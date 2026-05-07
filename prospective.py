import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from data_loader import (
    load_all, apply_theme, metric_card, section_header,
    GREEN, GREEN_PALE, RED, RED_PALE, AMBER, AMBER_PALE,
    GRAY, BLUE, BLUE_PALE, SEG_COLORS, SEG_LABELS, AXIS_BASE
)
from cate_engine import compute_cate, compute_variable_importance

@st.cache_data
def get_cate():
    return compute_cate()

@st.cache_data
def get_vi():
    return compute_variable_importance()

def render():
    users, campaigns, exposures, daily = load_all()
    cate_df = get_cate()

    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <h1 style="font-size:22px;font-weight:600;color:#1A252F;margin:0;">
            Campaign planner</h1>
        <p style="font-size:14px;color:{GRAY};margin:4px 0 0;">
            Users ranked by individual causal lift (CATE) — estimated via Causal Forest on holdout experiment data.</p>
    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # CAMPAIGN SELECTOR + PARAMETERS
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("Campaign parameters", expanded=True):
        st.markdown(
            f'<div style="font-size:12px;color:{GRAY};margin-bottom:14px;">'
            f'Select the reference campaign whose CATE estimates will rank the target list. '
            f'Adjust filters to refine who qualifies.</div>',
            unsafe_allow_html=True
        )

        pc1, pc2 = st.columns(2)

        with pc1:
            st.markdown(f'<div style="font-size:12px;font-weight:600;color:#2C3E50;margin-bottom:6px;">Budget & campaign</div>', unsafe_allow_html=True)

            camp_options = {
                f"{r.campaign_name}": r.campaign_id
                for _, r in campaigns.iterrows()
            }
            selected_camp = st.selectbox(
                "Reference campaign",
                list(camp_options.keys()),
                help="CATE estimates are drawn from this campaign's holdout experiment. Users who weren't in this campaign are scored using the nearest cross-campaign CATE."
            )
            ref_camp_id = camp_options[selected_camp]

            budget = st.slider(
                "Total campaign budget",
                min_value=5000, max_value=80000, value=30000, step=1000,
                format="$%d",
            )

            target_segs = st.multiselect(
                "User segments to target",
                options=list(SEG_LABELS.keys()),
                default=["dormant_high_value", "casual"],
                format_func=lambda x: SEG_LABELS[x],
            )

        with pc2:
            st.markdown(f'<div style="font-size:12px;font-weight:600;color:#2C3E50;margin-bottom:6px;">Targeting filters</div>', unsafe_allow_html=True)

            min_cate = st.slider(
                "Minimum causal lift (CATE) per user",
                min_value=-1.0, max_value=5.0, value=0.0, step=0.1,
                format="$%.1f/day",
                help=(
                    "Only include users whose estimated individual causal lift exceeds this threshold. "
                    "CATE = expected NGR/day increase caused by the promo, not just correlated with it. "
                    "Setting to $0 excludes users estimated to have zero or negative causal response."
                )
            )
            st.markdown(
                f'<div style="font-size:11px;color:{GRAY};margin:-8px 0 14px;">'
                f'Only users with estimated causal lift ≥ ${min_cate:.1f}/day</div>',
                unsafe_allow_html=True
            )

            max_propensity = st.slider(
                "Exclude users who would bet anyway",
                min_value=0.4, max_value=1.0, value=0.7, step=0.05, format="%.2f",
                help="Filters out always-on bettors whose CATE is likely near zero regardless."
            )
            st.markdown(
                f'<div style="font-size:11px;color:{GRAY};margin:-8px 0 14px;">'
                f'Excluding users with >{max_propensity:.0%} organic bet probability</div>',
                unsafe_allow_html=True
            )

            reliable_only = st.checkbox(
                "Only show reliable CATE estimates",
                value=False,
                help="Reliable = estimated from ≥2 matched holdout neighbors. Unreliable estimates have wider confidence intervals."
            )

    # ── MERGE CATE ONTO USERS ─────────────────────────────────────────────────
    # Use reference campaign CATEs; for users not in that campaign,
    # fall back to their best available CATE from any campaign
    ref_cate = cate_df[cate_df.campaign_id == ref_camp_id][
        ["user_id","cate","cate_ci_low","cate_ci_high","reliable","offer_amount"]
    ].copy()

    # Users not in ref campaign — use mean CATE across campaigns
    fallback = (cate_df.groupby("user_id")
                .agg(cate=("cate","mean"),
                     cate_ci_low=("cate_ci_low","mean"),
                     cate_ci_high=("cate_ci_high","mean"),
                     reliable=("reliable","min"),
                     offer_amount=("offer_amount","mean"))
                .reset_index())

    all_cate = ref_cate.copy()
    missing  = users[~users.user_id.isin(ref_cate.user_id)]["user_id"]
    fallback_sub = fallback[fallback.user_id.isin(missing)]
    all_cate = pd.concat([all_cate, fallback_sub], ignore_index=True)

    scored = users.merge(all_cate, on="user_id", how="left")
    scored["cate"]     = scored["cate"].fillna(0)
    scored["reliable"] = scored["reliable"].fillna(False).astype(bool)

    # ── FILTER ────────────────────────────────────────────────────────────────
    segs_to_use = target_segs if target_segs else list(SEG_LABELS.keys())
    filtered = scored[
        (scored.cate             >= min_cate) &
        (scored.propensity_score <= max_propensity) &
        (scored.segment_label.isin(segs_to_use))
    ].copy()

    if reliable_only:
        filtered = filtered[filtered.reliable]

    filtered = filtered.sort_values("cate", ascending=False)

    # ── BUDGET ALLOCATION ─────────────────────────────────────────────────────
    filtered["alloc_offer"]      = filtered["optimal_offer_size"].fillna(30).clip(upper=100)
    filtered["cumulative_spend"] = filtered["alloc_offer"].cumsum()
    funded   = filtered[filtered["cumulative_spend"] <= budget].copy()
    waitlist = filtered[filtered["cumulative_spend"] >  budget].copy()

    actual_spend = funded["alloc_offer"].sum()
    # Expected NGR = CATE × 14 days × number of funded users
    expected_ngr = (funded["cate"] * 14).sum()
    roi = expected_ngr / actual_spend if actual_spend else 0

    # ── KPI ROW ───────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(metric_card(
            "Users to target", f"{len(funded):,}",
            delta=f"{len(waitlist)} below budget cutoff", delta_good=False,
        ), unsafe_allow_html=True)
    with k2:
        st.markdown(metric_card(
            "Budget allocated", f"${actual_spend:,.0f}",
            delta=f"${budget - actual_spend:,.0f} unspent",
            delta_good=(budget - actual_spend) < budget * 0.1,
        ), unsafe_allow_html=True)
    with k3:
        st.markdown(metric_card(
            "Expected incremental NGR", f"${expected_ngr:,.0f}",
            note="CATE × 14-day window",
        ), unsafe_allow_html=True)
    with k4:
        st.markdown(metric_card(
            "Projected ROI", f"{roi:.1f}x",
            delta="vs 1.0x break-even", delta_good=roi >= 1.5,
        ), unsafe_allow_html=True)

    st.markdown("<div style='margin:20px 0 0;'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TARGET LIST
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "Target list",
        f"{len(funded)} users ranked by individual causal lift (CATE) · ready to export"
    )

    if funded.empty:
        st.warning("No users match the current filter criteria. Try relaxing the parameters.")
    else:
        display_cols = {
            "user_id":            "User ID",
            "segment_label":      "Segment",
            "days_since_last_bet":"Days inactive",
            "primary_sport":      "Sport",
            "cate":               "Causal lift ($/day)",
            "cate_ci_low":        "CI low",
            "cate_ci_high":       "CI high",
            "propensity_score":   "Organic bet prob.",
            "reliable":           "Reliable",
            "alloc_offer":        "Recommended offer ($)",
        }

        tbl = funded[list(display_cols.keys())].rename(columns=display_cols).copy()
        tbl["Segment"]  = tbl["Segment"].map(SEG_LABELS).fillna(tbl["Segment"])
        tbl["Reliable"] = tbl["Reliable"].map({True: "✓", False: "~"})

        def color_cate(val):
            if not isinstance(val, float): return ""
            if val >= 2.0:  return f"background-color:{GREEN_PALE};color:{GREEN};"
            if val >= 0.5:  return f"background-color:#EBF5FB;color:{BLUE};"
            if val < 0:     return f"background-color:{RED_PALE};color:{RED};"
            return ""

        st.dataframe(
            tbl.style
               .format({
                   "Causal lift ($/day)": "${:+.2f}",
                   "CI low":              "${:.2f}",
                   "CI high":             "${:.2f}",
                   "Organic bet prob.":   "{:.0%}",
                   "Recommended offer ($)":"${:.0f}",
                   "Days inactive":        "{:.0f}",
               })
               .map(color_cate, subset=["Causal lift ($/day)"]),
            use_container_width=True,
            height=340,
        )

        csv = tbl.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇  Export {len(funded)} users as CSV",
            data=csv,
            file_name=f"dk_cate_target_list.csv",
            mime="text/csv",
        )

    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # TARGETING DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div style="margin:0 0 4px;">
        <h2 style="font-size:17px;font-weight:600;color:#1A252F;margin:0;">
            Targeting diagnostics</h2>
        <p style="font-size:13px;color:{GRAY};margin:4px 0 0;">
            Distribution of causal lift estimates and who is and isn't in this campaign.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

    # ── CATE DISTRIBUTION ─────────────────────────────────────────────────────
    section_header(
        "CATE distribution — causal lift across all users",
        "Each bar is a user's estimated individual causal lift. "
        "Filled = funded. Hollow = filtered out. Error bars = 95% CI."
    )

    fig_dist = go.Figure()

    # All users sorted by CATE
    scored_sorted = scored.sort_values("cate", ascending=False).reset_index(drop=True)
    scored_sorted["funded_flag"] = scored_sorted.user_id.isin(funded.user_id)
    scored_sorted["color"] = scored_sorted.apply(
        lambda r: (GREEN if r.funded_flag else GRAY), axis=1
    )
    scored_sorted["opacity"] = scored_sorted["funded_flag"].map({True: 0.85, False: 0.25})

    # Funded users bars
    f_sub = scored_sorted[scored_sorted.funded_flag]
    nf_sub = scored_sorted[~scored_sorted.funded_flag]

    if not f_sub.empty:
        fig_dist.add_trace(go.Bar(
            x=list(range(len(f_sub))),
            y=f_sub["cate"].values,
            error_y=dict(
                type="data",
                array=(f_sub["cate_ci_high"] - f_sub["cate"]).fillna(0).values,
                arrayminus=(f_sub["cate"] - f_sub["cate_ci_low"]).fillna(0).values,
                color=GRAY, thickness=0.8, width=0,
            ),
            marker_color=GREEN, marker_opacity=0.85,
            name="Funded",
            hovertemplate="<b>%{text}</b><br>CATE: $%{y:+.2f}/day<extra></extra>",
            text=f_sub["user_id"].values,
        ))

    if not nf_sub.empty:
        fig_dist.add_trace(go.Bar(
            x=list(range(len(f_sub), len(f_sub) + len(nf_sub))),
            y=nf_sub["cate"].values,
            marker_color=GRAY, marker_opacity=0.2,
            name="Not funded",
            hovertemplate="<b>%{text}</b><br>CATE: $%{y:+.2f}/day<extra></extra>",
            text=nf_sub["user_id"].values,
        ))

    fig_dist.add_hline(y=0, line_color=GRAY, line_width=1, line_dash="dot")
    fig_dist.add_hline(y=min_cate, line_color=AMBER, line_width=1, line_dash="dash",
                       annotation_text=f"Min CATE threshold: ${min_cate:.1f}",
                       annotation_font_size=10, annotation_font_color=AMBER)

    apply_theme(fig_dist, height=300)
    fig_dist.update_layout(
        xaxis_title="Users (sorted by CATE, high to low)",
        yaxis_title="Estimated causal lift ($/user/day)",
        yaxis=dict(**AXIS_BASE, tickprefix="$"),
        xaxis=dict(**AXIS_BASE, showticklabels=False),
        barmode="overlay", showlegend=True,
        legend=dict(font_size=10, orientation="h", y=-0.18, x=0),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── CATE BY SEGMENT ───────────────────────────────────────────────────────
    col_s, col_p = st.columns(2)

    with col_s:
        section_header(
            "Mean CATE by segment",
            "Which segments show the highest average causal lift?"
        )
        seg_cate = (scored[scored.reliable]
                    .groupby("segment_label")["cate"]
                    .agg(mean_cate="mean", sem="sem", count="count")
                    .reset_index()
                    .sort_values("mean_cate", ascending=True))
        seg_cate["label"] = seg_cate["segment_label"].map(SEG_LABELS).fillna("Unknown")
        seg_cate["color"] = seg_cate["segment_label"].map(SEG_COLORS).fillna(GRAY)
        seg_cate["ci"]    = seg_cate["sem"] * 1.96

        fig_s = go.Figure(go.Bar(
            x=seg_cate["mean_cate"],
            y=seg_cate["label"],
            orientation="h",
            marker_color=seg_cate["color"],
            error_x=dict(type="data", array=seg_cate["ci"].tolist(),
                         color=GRAY, thickness=1.5, width=5),
            text=[f"${v:+.2f}  (n={int(c)})"
                  for v, c in zip(seg_cate["mean_cate"], seg_cate["count"])],
            textposition="outside", textfont_size=11,
        ))
        fig_s.add_vline(x=0, line_color=GRAY, line_width=1)
        apply_theme(fig_s, height=280)
        fig_s.update_layout(
            xaxis=dict(**AXIS_BASE, tickprefix="$"),
            showlegend=False, yaxis_title="",
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with col_p:
        section_header(
            "CATE vs. propensity score",
            "Do low-propensity users actually respond more causally?"
        )
        plot_df = scored[scored.reliable].copy()
        plot_df["seg_label"] = plot_df["segment_label"].map(SEG_LABELS).fillna("Unknown")

        fig_p = go.Figure()
        for seg, grp in plot_df.groupby("segment_label"):
            fig_p.add_trace(go.Scatter(
                x=grp["propensity_score"],
                y=grp["cate"],
                mode="markers",
                marker=dict(size=8, color=SEG_COLORS.get(seg, GRAY),
                            opacity=0.75, line=dict(width=0.5, color="#fff")),
                name=SEG_LABELS.get(seg, seg),
                hovertemplate="<b>%{text}</b><br>Propensity: %{x:.2f}<br>CATE: $%{y:+.2f}<extra></extra>",
                text=grp["user_id"],
            ))

        # Trend line
        if len(plot_df) > 3:
            slope, intercept, *_ = linregress(plot_df["propensity_score"], plot_df["cate"])
            x_range = np.linspace(0, 1, 50)
            fig_p.add_trace(go.Scatter(
                x=x_range, y=intercept + slope * x_range,
                mode="lines",
                line=dict(color=GRAY, width=1.5, dash="dot"),
                name="Trend", showlegend=False,
            ))

        fig_p.add_hline(y=0, line_color=GRAY, line_width=1, line_dash="dot")
        apply_theme(fig_p, height=280)
        fig_p.update_layout(
            xaxis_title="Propensity score (organic bet probability)",
            yaxis_title="CATE ($/user/day)",
            xaxis=dict(**AXIS_BASE, tickformat=".0%"),
            yaxis=dict(**AXIS_BASE, tickprefix="$"),
            legend=dict(font_size=10),
        )
        st.plotly_chart(fig_p, use_container_width=True)

    # ── ALLOCATION AUDIT ──────────────────────────────────────────────────────
    section_header(
        "Historical allocation audit",
        "Have past campaigns systematically under-promoted certain segments? "
        "Controlled for betting volume — apples-to-apples."
    )

    seg_promo = (users.groupby("segment_label")["avg_promo_value_received"]
                 .mean().reset_index()
                 .sort_values("avg_promo_value_received", ascending=True))
    seg_promo["label"]   = seg_promo["segment_label"].map(SEG_LABELS).fillna("Unknown")
    seg_promo["color"]   = seg_promo["segment_label"].map(SEG_COLORS).fillna(GRAY)
    seg_ngr              = users.groupby("segment_label")["ngr_90d"].mean()
    seg_promo["ngr_avg"] = seg_promo["segment_label"].map(seg_ngr)

    if len(seg_promo) > 2:
        slope, intercept, *_ = linregress(seg_promo["ngr_avg"],
                                          seg_promo["avg_promo_value_received"])
        seg_promo["expected_promo"] = intercept + slope * seg_promo["ngr_avg"]
        seg_promo["residual"]       = (seg_promo["avg_promo_value_received"]
                                       - seg_promo["expected_promo"])
    else:
        seg_promo["residual"] = 0

    jc1, jc2 = st.columns(2)

    with jc1:
        fig_j1 = go.Figure(go.Bar(
            x=seg_promo["avg_promo_value_received"], y=seg_promo["label"],
            orientation="h", marker_color=seg_promo["color"],
            text=[f"${v:.0f}" for v in seg_promo["avg_promo_value_received"]],
            textposition="outside", textfont_size=11,
        ))
        apply_theme(fig_j1, "Avg promo value received historically ($)", height=260)
        fig_j1.update_layout(xaxis=dict(**AXIS_BASE, tickprefix="$"),
                              showlegend=False, margin=dict(l=0, r=40, t=40, b=0))
        st.plotly_chart(fig_j1, use_container_width=True)

    with jc2:
        if seg_promo["residual"].std() > 0:
            seg_promo["res_color"] = seg_promo["residual"].apply(
                lambda v: GREEN if v >= 0 else RED
            )
            fig_j2 = go.Figure(go.Bar(
                x=seg_promo["residual"], y=seg_promo["label"],
                orientation="h", marker_color=seg_promo["res_color"],
                text=[f"${v:+.0f}" for v in seg_promo["residual"]],
                textposition="outside", textfont_size=11,
            ))
            fig_j2.add_vline(x=0, line_color=GRAY, line_width=1)
            apply_theme(fig_j2, "Over/under-allocation vs. expected ($)", height=260)
            fig_j2.update_layout(xaxis=dict(**AXIS_BASE, tickprefix="$"),
                                  showlegend=False, margin=dict(l=0, r=40, t=40, b=0))
            st.plotly_chart(fig_j2, use_container_width=True)

            under = seg_promo[seg_promo["residual"] < -3]
            if not under.empty:
                names = ", ".join(under["label"].tolist())
                st.markdown(
                    f'<div style="font-size:12px;color:{RED};background:{RED_PALE};'
                    f'padding:8px 12px;border-radius:6px;margin-top:8px;">'
                    f'⚠ Systematic under-allocation: <b>{names}</b></div>',
                    unsafe_allow_html=True
                )
