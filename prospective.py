import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import linregress
from data_loader import (
    load_all, apply_theme, metric_card, section_header,
    GREEN, GREEN_LIGHT, GREEN_MID, GREEN_PALE, RED, RED_PALE,
    AMBER, AMBER_PALE, GRAY, GRAY_LIGHT, BLUE, BLUE_PALE,
    SEG_COLORS, SEG_LABELS, PROMO_COLORS, LAYOUT_BASE, AXIS_BASE
)

def render():
    users, campaigns, exposures, daily = load_all()

    st.markdown("""
    <div style="margin-bottom:24px;">
        <h1 style="font-size:22px;font-weight:600;color:#1A252F;margin:0;">
            Campaign planner</h1>
        <p style="font-size:14px;color:#7F8C8D;margin:4px 0 0;">
            Set your budget and targeting criteria — get a ranked list of users to contact.</p>
    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PLANNING CONTROLS — plain language
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("Campaign parameters", expanded=True):

        st.markdown(
            f'<div style="font-size:12px;color:{GRAY};margin-bottom:14px;">'
            f'Adjust the sliders to define who qualifies for this campaign. '
            f'The target list and projections update automatically.</div>',
            unsafe_allow_html=True
        )

        pc1, pc2 = st.columns(2)

        with pc1:
            st.markdown(
                f'<div style="font-size:12px;font-weight:600;color:#2C3E50;margin-bottom:6px;">'
                f'Budget & offer type</div>', unsafe_allow_html=True
            )
            budget = st.slider(
                "Total campaign budget",
                min_value=5000, max_value=80000, value=30000, step=1000,
                format="$%d",
                help="Total promo dollars for this campaign. Users are funded in ranked order until exhausted."
            )
            promo_type = st.selectbox(
                "Promo type",
                ["deposit_bonus", "parlay_boost", "free_bet", "odds_boost"],
                format_func=lambda x: x.replace("_", " ").title(),
            )

            st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:12px;font-weight:600;color:#2C3E50;margin-bottom:6px;">'
                f'Who to include</div>', unsafe_allow_html=True
            )
            target_segs = st.multiselect(
                "User segments to target",
                options=list(SEG_LABELS.keys()),
                default=["dormant_high_value", "casual"],
                format_func=lambda x: SEG_LABELS[x],
                help="Which behavioral segments to draw from. Dormant high-value users typically yield the best ROI."
            )

        with pc2:
            st.markdown(
                f'<div style="font-size:12px;font-weight:600;color:#2C3E50;margin-bottom:6px;">'
                f'Targeting filters</div>', unsafe_allow_html=True
            )

            max_propensity = st.slider(
                "Exclude users who would bet anyway",
                min_value=0.4, max_value=1.0, value=0.7, step=0.05, format="%.2f",
                help=(
                    "Filters out users who are likely to bet regardless of any promo — "
                    "so budget isn't wasted on people already planning to deposit. "
                    "0.7 means: exclude anyone with a >70% chance of betting without a nudge."
                )
            )
            st.markdown(
                f'<div style="font-size:11px;color:{GRAY};margin:-8px 0 14px;">'
                f'Excluding users with &gt;{max_propensity:.0%} organic bet probability</div>',
                unsafe_allow_html=True
            )

            min_churn = st.slider(
                "Minimum inactivity risk",
                min_value=0.0, max_value=0.6, value=0.1, step=0.05, format="%.2f",
                help=(
                    "Only target users who show some risk of going dormant. "
                    "Avoids spending promo dollars on fully active users who don't need a nudge. "
                    "0.1 = include anyone with at least a 10% chance of no bet in the next 30 days."
                )
            )
            st.markdown(
                f'<div style="font-size:11px;color:{GRAY};margin:-8px 0 14px;">'
                f'Including users with ≥{min_churn:.0%} chance of going inactive in 30 days</div>',
                unsafe_allow_html=True
            )

            min_roi_score = st.slider(
                "Minimum expected promo ROI per user",
                min_value=0.0, max_value=0.4, value=0.05, step=0.01, format="%.2f",
                help=(
                    "A 0–1 score that ranks users by expected incremental return per promo dollar. "
                    "It combines three signals: how likely they are to respond to an offer, "
                    "how unlikely they are to bet without one, and how at-risk they are of churning. "
                    "0.05 is a reasonable floor — below that, expected return rarely justifies the spend."
                )
            )
            st.markdown(
                f'<div style="font-size:11px;color:{GRAY};margin:-8px 0 14px;">'
                f'Only users whose expected ROI score ≥ {min_roi_score:.2f}</div>',
                unsafe_allow_html=True
            )

    # ── FILTER & RANK USERS ───────────────────────────────────────────────────
    segs_to_use = target_segs if target_segs else list(SEG_LABELS.keys())

    filtered = users[
        (users.composite_score  >= min_roi_score) &
        (users.propensity_score <= max_propensity) &
        (users.churn_risk_30d   >= min_churn) &
        (users.segment_label.isin(segs_to_use))
    ].copy().sort_values("composite_score", ascending=False)

    filtered["alloc_offer"]      = filtered["optimal_offer_size"].clip(
        upper=max(budget / max(len(filtered), 1) * 2, 10)
    )
    filtered["cumulative_spend"] = filtered["alloc_offer"].cumsum()
    funded   = filtered[filtered["cumulative_spend"] <= budget].copy()
    waitlist = filtered[filtered["cumulative_spend"] >  budget].copy()

    actual_spend = funded["alloc_offer"].sum()
    expected_ngr = (
        funded["lift_probability"] *
        funded["ngr_90d"] / 90 *
        (1 - funded["propensity_score"]) * 14
    ).sum()
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
            note="Over 14-day campaign window",
        ), unsafe_allow_html=True)
    with k4:
        st.markdown(metric_card(
            "Projected ROI", f"{roi:.1f}x",
            delta="vs 1.0x break-even", delta_good=roi >= 1.5,
        ), unsafe_allow_html=True)

    st.markdown("<div style='margin:20px 0 0;'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TARGET LIST — immediately after KPIs
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "Target list",
        f"{len(funded)} users ranked by expected promo ROI · ready to export"
    )

    if funded.empty:
        st.warning("No users match the current filter criteria. Try relaxing the targeting parameters.")
    else:
        display_cols = {
            "user_id":            "User ID",
            "segment_label":      "Segment",
            "days_since_last_bet":"Days inactive",
            "primary_sport":      "Sport",
            "churn_risk_30d":     "Churn risk",
            "propensity_score":   "Organic bet prob.",
            "lift_probability":   "Promo response prob.",
            "composite_score":    "Expected ROI score",
            "alloc_offer":        "Recommended offer ($)",
        }

        tbl = funded[list(display_cols.keys())].rename(columns=display_cols).copy()
        tbl["Segment"] = tbl["Segment"].map(SEG_LABELS).fillna(tbl["Segment"])

        def color_roi(val):
            if not isinstance(val, float): return ""
            if val >= 0.15: return f"background-color:{GREEN_PALE};color:{GREEN};"
            if val >= 0.07: return f"background-color:{AMBER_PALE};color:{AMBER};"
            return f"background-color:{RED_PALE};color:{RED};"

        st.dataframe(
            tbl.style
               .format({
                   "Churn risk":           "{:.0%}",
                   "Organic bet prob.":    "{:.0%}",
                   "Promo response prob.": "{:.0%}",
                   "Expected ROI score":   "{:.3f}",
                   "Recommended offer ($)":"${:.0f}",
                   "Days inactive":        "{:.0f}",
               })
               .map(color_roi, subset=["Expected ROI score"]),
            use_container_width=True,
            height=340,
        )

        csv = tbl.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇  Export {len(funded)} users as CSV",
            data=csv,
            file_name=f"dk_target_list_{promo_type}.csv",
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
            Who is and isn't in this campaign — and whether historical spend has been allocated efficiently.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

    section_header(
        "Who's in the list — and who's excluded",
        "Every user plotted by organic bet probability vs. promo response probability. "
        "Filled = funded. Hollow = filtered out. Dot size = recommended offer amount."
    )

    all_display = users.copy()
    all_display["funded"] = all_display.user_id.isin(funded.user_id)
    fig_f = go.Figure()

    for seg, grp in all_display.groupby("segment_label"):
        color = SEG_COLORS.get(seg, GRAY)
        label = SEG_LABELS.get(seg, seg)
        grp_f  = grp[grp.funded]
        grp_nf = grp[~grp.funded]

        if not grp_f.empty:
            fig_f.add_trace(go.Scatter(
                x=grp_f.propensity_score,
                y=grp_f.lift_probability,
                mode="markers",
                marker=dict(
                    size=np.clip(grp_f.optimal_offer_size / 4, 6, 20),
                    color=color, opacity=0.85,
                    line=dict(width=1, color="#fff"),
                ),
                name=label,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Organic bet prob: %{x:.0%}<br>"
                    "Promo response prob: %{y:.0%}<br>"
                    "Churn risk: %{customdata:.0%}<br>"
                    "ROI score: %{meta:.3f}<extra></extra>"
                ),
                text=grp_f.user_id,
                customdata=grp_f.churn_risk_30d,
                meta=grp_f.composite_score,
            ))

        if not grp_nf.empty:
            fig_f.add_trace(go.Scatter(
                x=grp_nf.propensity_score,
                y=grp_nf.lift_probability,
                mode="markers",
                marker=dict(size=6, color=GRAY, opacity=0.2,
                            line=dict(width=0.5, color=GRAY), symbol="circle-open"),
                showlegend=False, hoverinfo="skip",
            ))

    fig_f.add_vline(
        x=max_propensity, line_color=RED, line_dash="dash", line_width=1,
        annotation_text=f"Excluded: organic prob > {max_propensity:.0%}",
        annotation_font_size=10, annotation_font_color=RED,
    )
    fig_f.add_annotation(x=0.15, y=0.82, text="★ Best targets",
                          font=dict(size=10, color=GREEN), showarrow=False,
                          bgcolor=GREEN_PALE, borderpad=3)
    fig_f.add_annotation(x=0.82, y=0.15, text="Skip — bet anyway",
                          font=dict(size=10, color=RED), showarrow=False,
                          bgcolor=RED_PALE, borderpad=3)

    apply_theme(fig_f, height=360)
    fig_f.update_layout(
        xaxis_title="Organic bet probability — likelihood to bet without any promo",
        yaxis_title="Promo response probability — likelihood of behavior change with an offer",
        legend=dict(font_size=10, orientation="h", y=-0.18, x=0),
        xaxis=dict(**AXIS_BASE, tickformat=".0%"),
        yaxis=dict(**AXIS_BASE, tickformat=".0%"),
    )
    st.plotly_chart(fig_f, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORICAL ALLOCATION AUDIT
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "Historical allocation audit",
        "Have past campaigns systematically under-promoted certain segments? "
        "Controlled for betting volume — apples-to-apples comparison."
    )

    seg_promo = (users.groupby("segment_label")["avg_promo_value_received"]
                 .mean().reset_index()
                 .sort_values("avg_promo_value_received", ascending=True))
    seg_promo["label"]   = seg_promo["segment_label"].map(SEG_LABELS).fillna("Unknown")
    seg_promo["color"]   = seg_promo["segment_label"].map(SEG_COLORS).fillna(GRAY)
    seg_ngr              = users.groupby("segment_label")["ngr_90d"].mean()
    seg_promo["ngr_avg"] = seg_promo["segment_label"].map(seg_ngr)

    if len(seg_promo) > 2:
        slope, intercept, *_ = linregress(seg_promo["ngr_avg"], seg_promo["avg_promo_value_received"])
        seg_promo["expected_promo"] = intercept + slope * seg_promo["ngr_avg"]
        seg_promo["residual"]       = seg_promo["avg_promo_value_received"] - seg_promo["expected_promo"]
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
        fig_j1.update_layout(
            xaxis=dict(**AXIS_BASE, tickprefix="$"),
            showlegend=False, margin=dict(l=0, r=40, t=40, b=0),
        )
        st.plotly_chart(fig_j1, use_container_width=True)
        st.markdown(
            f'<div style="font-size:11px;color:{GRAY};">'
            f'Raw average promo dollars received per user across all past campaigns.</div>',
            unsafe_allow_html=True
        )

    with jc2:
        if seg_promo["residual"].std() > 0:
            seg_promo["res_color"] = seg_promo["residual"].apply(lambda v: GREEN if v >= 0 else RED)
            fig_j2 = go.Figure(go.Bar(
                x=seg_promo["residual"], y=seg_promo["label"],
                orientation="h", marker_color=seg_promo["res_color"],
                text=[f"${v:+.0f}" for v in seg_promo["residual"]],
                textposition="outside", textfont_size=11,
            ))
            fig_j2.add_vline(x=0, line_color=GRAY, line_width=1)
            apply_theme(fig_j2, "Over/under-allocation vs. expected ($)", height=260)
            fig_j2.update_layout(
                xaxis=dict(**AXIS_BASE, tickprefix="$"),
                showlegend=False, margin=dict(l=0, r=40, t=40, b=0),
            )
            st.plotly_chart(fig_j2, use_container_width=True)
            st.markdown(
                f'<div style="font-size:11px;color:{GRAY};">'
                f'After controlling for each segment\'s NGR contribution — which segments are '
                f'receiving more or less promo than their value justifies. '
                f'Negative = systematically under-targeted.</div>',
                unsafe_allow_html=True
            )

            under = seg_promo[seg_promo["residual"] < -3]
            if not under.empty:
                names = ", ".join(under["label"].tolist())
                st.markdown(
                    f'<div style="font-size:12px;color:{RED};background:{RED_PALE};'
                    f'padding:8px 12px;border-radius:6px;margin-top:8px;">'
                    f'⚠ Systematic under-allocation detected: <b>{names}</b> — '
                    f'receiving less promo value than matched profiles in other segments.</div>',
                    unsafe_allow_html=True
                )
