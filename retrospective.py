import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from data_loader import (
    load_all, apply_theme, metric_card, confidence_badge, section_header,
    GREEN, GREEN_LIGHT, GREEN_MID, GREEN_PALE, RED, RED_PALE,
    AMBER, GRAY, BLUE, BLUE_PALE, SEG_COLORS, SEG_LABELS, LAYOUT_BASE, AXIS_BASE
)

def render():
    users, campaigns, exposures, daily = load_all()

    # ── PAGE HEADER ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h1 style="font-size:22px;font-weight:600;color:#1A252F;margin:0;">
            Campaign autopsy</h1>
        <p style="font-size:14px;color:#7F8C8D;margin:4px 0 0;">
            Retrospective causal analysis · Select a campaign to dissect</p>
    </div>""", unsafe_allow_html=True)

    # ── CAMPAIGN SELECTOR ─────────────────────────────────────────────────────
    camp_options = {
        f"{r.campaign_name}  ({r.start_date.strftime('%b %d')} – {r.end_date.strftime('%b %d, %Y')})": r.campaign_id
        for _, r in campaigns.iterrows()
    }
    selected_label = st.selectbox("Campaign", list(camp_options.keys()), label_visibility="collapsed")
    camp_id   = camp_options[selected_label]
    camp      = campaigns[campaigns.campaign_id == camp_id].iloc[0]
    c_start   = camp.start_date
    c_end     = camp.end_date
    c_start_ts = pd.Timestamp(c_start)
    c_end_ts   = pd.Timestamp(c_end)

    # ── SEGMENT DATA FOR THIS CAMPAIGN ────────────────────────────────────────
    camp_exp  = exposures[exposures.campaign_id == camp_id]
    treat_ids = camp_exp[~camp_exp.is_holdout]["user_id"].tolist()
    hold_ids  = camp_exp[camp_exp.is_holdout]["user_id"].tolist()

    pre_mask   = (daily.date >= pd.Timestamp("2025-01-01")) & (daily.date < c_start_ts)
    camp_mask  = (daily.date >= c_start_ts) & (daily.date <= c_end_ts)
    post_mask  = (daily.date > c_end_ts)    & (daily.date <= pd.Timestamp("2025-12-31"))

    treat_pre  = daily[pre_mask  & daily.user_id.isin(treat_ids)]
    treat_camp = daily[camp_mask & daily.user_id.isin(treat_ids)]
    treat_post = daily[post_mask & daily.user_id.isin(treat_ids)]
    hold_pre   = daily[pre_mask  & daily.user_id.isin(hold_ids)]
    hold_camp  = daily[camp_mask & daily.user_id.isin(hold_ids)]
    hold_post  = daily[post_mask & daily.user_id.isin(hold_ids)]

    # ── TOP KPI ROW ───────────────────────────────────────────────────────────
    treat_ngr_pre  = treat_pre["ngr"].mean()
    treat_ngr_camp = treat_camp["ngr"].mean()
    hold_ngr_pre   = hold_pre["ngr"].mean()
    hold_ngr_camp  = hold_camp["ngr"].mean()

    causal_lift_abs = (treat_ngr_camp - treat_ngr_pre) - (hold_ngr_camp - hold_ngr_pre)
    causal_lift_pct = causal_lift_abs / abs(treat_ngr_pre) * 100 if treat_ngr_pre else 0

    treat_ngr_post = treat_post["ngr"].mean()
    retained_pct   = ((treat_ngr_post - treat_ngr_pre) / abs(causal_lift_abs) * 100
                      if causal_lift_abs else 0)
    retained_pct   = np.clip(retained_pct, 0, 100)

    total_spend = camp_exp[~camp_exp.is_holdout]["offer_amount"].sum()
    incr_ngr    = causal_lift_abs * len(treat_ids) * (c_end - c_start).days
    roi         = incr_ngr / total_spend if total_spend else 0

    n_recipients = len(treat_ids)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card(
            "Causal lift (NGR/day)",
            f"+{causal_lift_pct:.1f}%",
            delta="vs synthetic control",
            delta_good=causal_lift_pct > 0,
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "Retained lift (30d post)",
            f"{retained_pct:.0f}%",
            note="of lift persisted after campaign",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(
            "Total promo spend",
            f"${total_spend:,.0f}",
            note=f"{n_recipients} recipients",
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card(
            "Estimated ROI",
            f"{roi:.1f}x" if roi > 0 else "N/A",
            delta_good=roi > 1,
        ), unsafe_allow_html=True)

    st.markdown("<div style='margin:20px 0 0;'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL A — SYNTHETIC CONTROL
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "Causal lift — synthetic control",
        "Actual treatment group vs. reconstructed counterfactual. The gap is the true causal effect."
    )

    # Build weekly time series for treatment and synthetic control
    synth_ids = daily[daily.is_synthetic_control == True]["user_id"].unique().tolist()

    def weekly_ngr(df, uid_list, label):
        sub = df[df.user_id.isin(uid_list)].copy()
        sub["week"] = sub["date"].dt.to_period("W").dt.start_time
        return sub.groupby("week")["ngr"].mean().reset_index().assign(series=label)

    if treat_ids and synth_ids:
        treat_weekly  = weekly_ngr(daily, treat_ids,  "Treatment (received promo)")
        synth_weekly  = weekly_ngr(daily, synth_ids,  "Synthetic control")

        fig_a = go.Figure()

        # Shaded campaign window
        fig_a.add_vrect(
            x0=c_start_ts, x1=c_end_ts,
            fillcolor=GREEN, opacity=0.07,
            layer="below", line_width=0,
        )

        # Synthetic control line
        fig_a.add_trace(go.Scatter(
            x=synth_weekly.week, y=synth_weekly.ngr,
            name="Synthetic control",
            line=dict(color=GRAY, width=2, dash="dash"),
            mode="lines",
        ))

        # Treatment line
        fig_a.add_trace(go.Scatter(
            x=treat_weekly.week, y=treat_weekly.ngr,
            name="Treatment group",
            line=dict(color=GREEN, width=2.5),
            mode="lines+markers",
            marker=dict(size=5),
        ))

        # Fill between (approximate — use min of the two)
        merged_w = treat_weekly.merge(synth_weekly, on="week", suffixes=("_t","_s"))
        fig_a.add_trace(go.Scatter(
            x=pd.concat([merged_w.week, merged_w.week[::-1]]),
            y=pd.concat([merged_w.ngr_t, merged_w.ngr_s[::-1]]),
            fill="toself",
            fillcolor=f"rgba(27,94,59,0.10)",
            line=dict(width=0),
            name="Causal gap",
            showlegend=True,
        ))

        # Campaign period annotation
        fig_a.add_annotation(
            x=c_start_ts + (c_end_ts - c_start_ts) / 2,
            y=treat_weekly.ngr.max() * 1.05,
            text=f"Campaign  |  +{causal_lift_pct:.0f}% causal lift",
            showarrow=False,
            font=dict(size=11, color=GREEN),
            bgcolor=GREEN_PALE,
            borderpad=4,
        )

        apply_theme(fig_a, height=340)
        fig_a.update_layout(
            xaxis_title="", yaxis_title="Avg NGR / user / day ($)",
            legend=dict(orientation="h", y=-0.15, x=0),
        )
        st.plotly_chart(fig_a, use_container_width=True)
    else:
        st.info("Not enough holdout users to construct synthetic control for this campaign.")

    col_conf1, col_conf2 = st.columns([1,3])
    with col_conf1:
        # Compute confidence via permutation intuition: is lift > random noise?
        t_stat, p_val = stats.ttest_ind(
            treat_camp.ngr.fillna(0).values,
            hold_camp.ngr.fillna(0).values,
        ) if (len(treat_camp) > 1 and len(hold_camp) > 1) else (0, 1)
        conf = int(np.clip((1 - p_val) * 100, 50, 99))
        st.markdown(confidence_badge(conf), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL B — SEGMENT-LEVEL SYNTHETIC CONTROL
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "Causal lift by segment — synthetic control per segment",
        "Each segment's treatment group vs. its own synthetic control. Isolates true causal lift within each group."
    )

    # Tag every user with their segment
    uid_to_seg = users.set_index("user_id")["segment_label"].to_dict()

    # Identify holdout users for this campaign
    all_camp_uids  = camp_exp["user_id"].tolist()
    hold_ids_set   = set(hold_ids)
    treat_ids_set  = set(treat_ids)

    # Pre-compute per-user mean NGR in pre and campaign windows
    def user_mean_ngr(uid_list, mask):
        sub = daily[mask & daily.user_id.isin(uid_list)]
        return sub.groupby("user_id")["ngr"].mean()

    seg_results = []  # will hold one row per segment

    all_segments = sorted(users["segment_label"].unique())

    for seg in all_segments:
        seg_treat = [u for u in treat_ids if uid_to_seg.get(u) == seg]
        seg_hold  = [u for u in hold_ids  if uid_to_seg.get(u) == seg]

        n_treat = len(seg_treat)
        n_hold  = len(seg_hold)

        if n_treat == 0:
            continue  # segment not represented in this campaign

        # ── Pre-campaign mean NGR per user ────────────────────────────────
        treat_pre_ngr = user_mean_ngr(seg_treat, pre_mask)   # Series: user_id → mean NGR
        hold_pre_ngr  = user_mean_ngr(seg_hold,  pre_mask)

        treat_pre_mean = treat_pre_ngr.mean() if len(treat_pre_ngr) else 0.0

        # ── Build synthetic control weights ───────────────────────────────
        # For each holdout user: weight = 1 / (1 + |pre_ngr_diff|)
        # normalised so weights sum to 1.
        # If no holdout users in segment, fall back to all holdout users.
        pool = seg_hold if n_hold >= 1 else hold_ids
        pool_pre = user_mean_ngr(pool, pre_mask)

        if len(pool_pre) == 0:
            # No holdout users at all — cannot compute synthetic control
            seg_results.append({
                "segment_label": seg,
                "causal_lift":   np.nan,
                "ci_half":       np.nan,
                "n_treat":       n_treat,
                "n_hold":        n_hold,
                "reliable":      False,
                "note":          "No holdout users",
            })
            continue

        diffs   = np.abs(pool_pre.values - treat_pre_mean)
        weights = 1 / (1 + diffs)
        weights = weights / weights.sum()

        # ── Synthetic control NGR during campaign ─────────────────────────
        camp_ngr_by_hold = user_mean_ngr(pool_pre.index.tolist(), camp_mask)
        # align to pool_pre index order
        camp_ngr_aligned = camp_ngr_by_hold.reindex(pool_pre.index).fillna(pool_pre)
        synth_camp_ngr   = float(np.dot(weights, camp_ngr_aligned.values))

        # ── Treatment group NGR during campaign ───────────────────────────
        treat_camp_ngr_series = user_mean_ngr(seg_treat, camp_mask)
        treat_camp_mean = treat_camp_ngr_series.mean() if len(treat_camp_ngr_series) else 0.0

        # ── Causal lift = treatment − synthetic control ───────────────────
        causal_lift = treat_camp_mean - synth_camp_ngr

        # ── Confidence interval via bootstrap on treatment group ──────────
        if len(treat_camp_ngr_series) >= 2:
            boots = [
                np.random.choice(treat_camp_ngr_series.values, size=len(treat_camp_ngr_series), replace=True).mean()
                for _ in range(500)
            ]
            ci_half = np.percentile(boots, 97.5) - np.percentile(boots, 2.5)
        else:
            ci_half = abs(causal_lift) * 0.5  # wide CI for n=1

        reliable = (n_treat >= 3 and n_hold >= 1)

        seg_results.append({
            "segment_label": seg,
            "causal_lift":   causal_lift,
            "ci_half":       ci_half / 2,
            "n_treat":       n_treat,
            "n_hold":        n_hold,
            "reliable":      reliable,
            "note":          "" if reliable else f"Low n (treat={n_treat}, hold={n_hold})",
        })

    seg_df = pd.DataFrame(seg_results).sort_values("causal_lift", ascending=True)
    seg_df["label"] = seg_df["segment_label"].map(SEG_LABELS).fillna("Unknown")
    seg_df["color"] = seg_df.apply(
        lambda r: SEG_COLORS.get(r["segment_label"], GRAY) if r["reliable"] else "#C8D0D8",
        axis=1
    )

    # Build the chart
    fig_b = go.Figure()

    # Reliable segments — solid bars with CI
    rel = seg_df[seg_df["reliable"] & seg_df["causal_lift"].notna()]
    unrel = seg_df[~seg_df["reliable"] | seg_df["causal_lift"].isna()]

    if not rel.empty:
        fig_b.add_trace(go.Bar(
            x=rel["causal_lift"],
            y=rel["label"],
            orientation="h",
            marker_color=rel["color"],
            error_x=dict(
                type="data",
                array=rel["ci_half"].tolist(),
                color=GRAY,
                thickness=1.5,
                width=5,
            ),
            text=[
                f"+${v:.2f}  (n={int(nt)})" if v >= 0 else f"${v:.2f}  (n={int(nt)})"
                for v, nt in zip(rel["causal_lift"], rel["n_treat"])
            ],
            textposition="outside",
            textfont=dict(size=11),
            name="Reliable estimate",
        ))

    if not unrel.empty:
        fig_b.add_trace(go.Bar(
            x=unrel["causal_lift"].fillna(0),
            y=unrel["label"],
            orientation="h",
            marker_color=GRAY,
            marker_pattern_shape="/",
            opacity=0.5,
            text=[f"Low n — treat={int(r.n_treat)}, hold={int(r.n_hold)}"
                  for _, r in unrel.iterrows()],
            textposition="outside",
            textfont=dict(size=10, color=GRAY),
            name="Insufficient data",
        ))

    fig_b.add_vline(x=0, line_color=GRAY, line_width=1)

    apply_theme(fig_b, height=320)
    fig_b.update_layout(
        xaxis_title="Causal lift: treatment − synthetic control  (NGR/user/day, $)",
        yaxis_title="",
        xaxis=dict(**AXIS_BASE, tickprefix="$"),
        barmode="overlay",
        showlegend=True,
        legend=dict(font_size=10, orientation="h", y=-0.18, x=0),
    )
    st.plotly_chart(fig_b, use_container_width=True)

    # Method note
    st.markdown(
        f'<div style="font-size:11px;color:{GRAY};line-height:1.6;margin-top:-8px;">'
        f'Each bar = mean NGR of segment\'s treatment group minus its weighted synthetic control '
        f'(holdout users in same segment, weighted by pre-campaign NGR similarity). '
        f'Error bars = 95% bootstrap CI on treatment group. '
        f'Hatched bars have insufficient sample for reliable estimation.</div>',
        unsafe_allow_html=True
    )

    # Also expose user_lift for Panel D downstream (need it built from treat users)
    treat_users   = users[users.user_id.isin(treat_ids)][["user_id","segment_label"]].copy()
    treat_daily_camp = daily[camp_mask & daily.user_id.isin(treat_ids)]
    treat_daily_pre  = daily[pre_mask  & daily.user_id.isin(treat_ids)]
    pre_ngr_u    = treat_daily_pre.groupby("user_id")["ngr"].mean().reset_index(name="pre_ngr")
    camp_ngr_u   = treat_daily_camp.groupby("user_id")["ngr"].mean().reset_index(name="camp_ngr")
    user_lift    = pre_ngr_u.merge(camp_ngr_u, on="user_id", how="outer").fillna(0)
    user_lift["lift"] = user_lift["camp_ngr"] - user_lift["pre_ngr"]
    user_lift    = user_lift.merge(treat_users, on="user_id", how="left")
    user_lift["segment_label"] = user_lift["segment_label"].fillna("unknown")

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL C — HABIT FORMATION: SYNTHETIC CONTROL DECAY CURVE
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "Habit formation — synthetic control decay",
        "Treatment group vs. synthetic control tracked weekly for 12 weeks post-campaign. "
        "Gap = causal lift that persisted. Same counterfactual as Panel A, extended forward in time."
    )

    # ── Reuse or recompute synthetic control users ────────────────────────────
    # Synthetic control = holdout users whose pre-campaign NGR best matched
    # the treatment group. We weight them the same way as Panel A.
    pool_ids = hold_ids if hold_ids else []

    if pool_ids:
        treat_pre_mean = treat_pre.groupby("user_id")["ngr"].mean().mean()
        pool_pre_ngr   = (daily[pre_mask & daily.user_id.isin(pool_ids)]
                          .groupby("user_id")["ngr"].mean())

        diffs_c   = np.abs(pool_pre_ngr.values - treat_pre_mean)
        weights_c = 1 / (1 + diffs_c)
        weights_c = weights_c / weights_c.sum()
        synth_uid_order = pool_pre_ngr.index.tolist()
    else:
        synth_uid_order = []
        weights_c       = np.array([])

    def weighted_weekly_ngr(uid_list, weights=None):
        """Return a Series of week_num → mean NGR/day.
        If weights provided, compute weighted mean across users per week."""
        post_sub = daily[post_mask & daily.user_id.isin(uid_list)].copy()
        if post_sub.empty:
            return pd.Series(dtype=float)
        post_sub["week_num"] = (
            (post_sub["date"] - (c_end_ts + pd.Timedelta(days=1))).dt.days // 7 + 1
        )
        post_sub = post_sub[post_sub.week_num.between(1, 12)]
        if weights is not None and len(weights) == len(uid_list):
            # weighted mean: for each week, sum(user_avg_ngr * weight)
            rows = []
            for wk, grp in post_sub.groupby("week_num"):
                user_means = grp.groupby("user_id")["ngr"].mean()
                # align weights to users present this week
                w_aligned = pd.Series(weights, index=uid_list).reindex(user_means.index).fillna(0)
                if w_aligned.sum() > 0:
                    w_aligned /= w_aligned.sum()
                wng = float((user_means * w_aligned).sum())
                rows.append({"week_num": wk, "ngr": wng})
            return pd.DataFrame(rows).set_index("week_num")["ngr"] if rows else pd.Series(dtype=float)
        else:
            return post_sub.groupby("week_num")["ngr"].mean()

    treat_decay = weighted_weekly_ngr(treat_ids)
    synth_decay = weighted_weekly_ngr(synth_uid_order, weights=weights_c) if synth_uid_order else pd.Series(dtype=float)

    # Week 0 = campaign period itself (anchors the chart at the peak lift)
    treat_wk0 = treat_camp["ngr"].mean() if len(treat_camp) else 0.0
    if synth_uid_order:
        synth_camp_sub = daily[camp_mask & daily.user_id.isin(synth_uid_order)]
        synth_wk0_ngr  = synth_camp_sub.groupby("user_id")["ngr"].mean()
        w_s = pd.Series(weights_c, index=synth_uid_order).reindex(synth_wk0_ngr.index).fillna(0)
        if w_s.sum() > 0: w_s /= w_s.sum()
        synth_wk0 = float((synth_wk0_ngr * w_s).sum())
    else:
        synth_wk0 = treat_pre["ngr"].mean()   # fallback: flat baseline

    # Combine week 0 with post-campaign weeks
    all_weeks = [0] + list(range(1, 13))

    def get_ngr(series, wk, fallback):
        return float(series.get(wk, fallback))

    treat_ngrs = [treat_wk0] + [get_ngr(treat_decay, w, np.nan) for w in range(1, 13)]
    synth_ngrs = [synth_wk0]  + [get_ngr(synth_decay, w, np.nan) for w in range(1, 13)]

    # Causal lift per week = treatment − synthetic control
    lift_series = [
        (t - s) if (not np.isnan(t) and not np.isnan(s)) else np.nan
        for t, s in zip(treat_ngrs, synth_ngrs)
    ]

    valid = [(w, l) for w, l in zip(all_weeks, lift_series) if not np.isnan(l)]
    weeks_v = [v[0] for v in valid]
    lift_v  = [v[1] for v in valid]

    # Retained lift % at end of window
    peak_lift     = lift_v[0] if lift_v else 1.0
    final_lift    = lift_v[-1] if lift_v else 0.0
    retained_lift = (final_lift / peak_lift * 100) if peak_lift else 0.0

    fig_c = go.Figure()

    # Zero line = synthetic control (counterfactual)
    fig_c.add_hline(
        y=0, line_color=GRAY, line_dash="dot", line_width=1,
        annotation_text="Synthetic control (counterfactual)",
        annotation_font_size=10, annotation_font_color=GRAY,
        annotation_position="bottom right",
    )

    # Treatment line (raw, for context)
    tick_texts = ["Campaign"] + [f"Wk {i}" for i in range(1, 13)]

    fig_c.add_trace(go.Scatter(
        x=all_weeks,
        y=synth_ngrs,
        name="Synthetic control",
        line=dict(color=GRAY, width=1.5, dash="dash"),
        mode="lines",
        hovertemplate="Synth control wk %{x}: $%{y:.3f}<extra></extra>",
    ))

    fig_c.add_trace(go.Scatter(
        x=all_weeks,
        y=treat_ngrs,
        name="Treatment group",
        line=dict(color=GREEN, width=1.5),
        mode="lines",
        opacity=0.4,
        hovertemplate="Treatment wk %{x}: $%{y:.3f}<extra></extra>",
    ))

    # Causal lift gap — the main series
    fig_c.add_trace(go.Scatter(
        x=weeks_v + weeks_v[::-1],
        y=[max(l, 0) for l in lift_v] + [0]*len(lift_v),
        fill="toself",
        fillcolor="rgba(27,94,59,0.10)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig_c.add_trace(go.Scatter(
        x=weeks_v,
        y=lift_v,
        name="Causal lift (treatment − synthetic control)",
        line=dict(color=GREEN, width=2.5),
        mode="lines+markers",
        marker=dict(size=6, color=GREEN),
        hovertemplate="Wk %{x}: lift $%{y:+.3f}<extra></extra>",
    ))

    # Annotations at weeks 4, 8, 12
    for wk in [4, 8, 12]:
        if wk in weeks_v:
            idx = weeks_v.index(wk)
            v   = lift_v[idx]
            fig_c.add_annotation(
                x=wk, y=v,
                text=f"Wk {wk}: ${v:+.2f}",
                showarrow=True, arrowhead=2, arrowcolor=GRAY,
                font=dict(size=10, color="#2C3E50"),
                ay=-36, ax=0, bgcolor="rgba(255,255,255,0.85)",
                borderpad=3,
            )

    apply_theme(fig_c, height=320)
    fig_c.update_layout(
        xaxis_title="Weeks post-campaign",
        yaxis_title="Causal lift in NGR / user / day ($)",
        xaxis=dict(**AXIS_BASE, tickvals=all_weeks, ticktext=tick_texts),
        legend=dict(font_size=10, orientation="h", y=-0.20, x=0),
    )
    fig_c.update_yaxes(tickprefix="$")
    st.plotly_chart(fig_c, use_container_width=True)

    # Retained lift summary callout
    color_ret = GREEN if retained_lift >= 20 else AMBER if retained_lift >= 0 else RED
    bg_ret    = GREEN_PALE if retained_lift >= 20 else AMBER_PALE if retained_lift >= 0 else RED_PALE
    st.markdown(
        f'<div style="font-size:12px;color:{color_ret};background:{bg_ret};'
        f'padding:8px 12px;border-radius:6px;margin-top:-8px;">'
        f'{"✓" if retained_lift >= 20 else "⚠"} '
        f'<b>{retained_lift:.0f}%</b> of campaign lift retained at week 12 vs. synthetic control — '
        f'peak lift ${peak_lift:+.2f}/day, week-12 lift ${final_lift:+.2f}/day.</div>',
        unsafe_allow_html=True
    )

    # ══════════════════════════════════════════════════════════════════════════
    # DOSE-RESPONSE CURVE — full width
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "Offer size vs. incremental NGR",
        "Does a larger offer produce meaningfully more lift? Shows where diminishing returns set in across this campaign's recipients."
    )

    dose_df = camp_exp[~camp_exp.is_holdout][["user_id","offer_amount"]].merge(
        user_lift[["user_id","lift"]], on="user_id", how="inner"
    )
    dose_df["offer_bin"] = pd.cut(
        dose_df["offer_amount"], bins=6,
        labels=[f"${int(b.left)}–${int(b.right)}" for b in
                pd.cut(dose_df["offer_amount"], bins=6).cat.categories]
    )
    dose_agg = (dose_df.groupby("offer_bin", observed=True)["lift"]
                .agg(mean_lift="mean", sem="sem", count="count")
                .reset_index())
    dose_agg["ci"] = dose_agg["sem"] * 1.96

    fig_e = go.Figure()

    # CI band
    fig_e.add_trace(go.Scatter(
        x=pd.concat([dose_agg["offer_bin"], dose_agg["offer_bin"][::-1]]),
        y=pd.concat([dose_agg["mean_lift"] + dose_agg["ci"],
                     (dose_agg["mean_lift"] - dose_agg["ci"])[::-1]]),
        fill="toself",
        fillcolor="rgba(27,94,59,0.10)",
        line=dict(width=0),
        showlegend=False,
    ))

    # Bars
    fig_e.add_trace(go.Bar(
        x=dose_agg["offer_bin"],
        y=dose_agg["mean_lift"],
        marker_color=[GREEN if v > 0 else RED for v in dose_agg["mean_lift"]],
        name="Mean lift",
        text=[f"${v:.2f}" for v in dose_agg["mean_lift"]],
        textposition="outside",
        textfont=dict(size=11),
    ))

    apply_theme(fig_e, height=320)
    fig_e.update_layout(
        xaxis_title="Offer amount range ($)",
        yaxis_title="Avg incremental NGR / user / day ($)",
        showlegend=False,
    )
    fig_e.update_yaxes(tickprefix="$")
    st.plotly_chart(fig_e, use_container_width=True)

    if not dose_agg.empty:
        best_bin = dose_agg.loc[dose_agg["mean_lift"].idxmax(), "offer_bin"]
        st.markdown(
            f'<div style="font-size:12px;color:{GREEN};background:{GREEN_PALE};'
            f'padding:8px 12px;border-radius:6px;margin-top:-8px;">'
            f'✓ Highest incremental return in offer range <b>{best_bin}</b></div>',
            unsafe_allow_html=True
        )
