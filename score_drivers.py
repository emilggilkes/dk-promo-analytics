import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_loader import (
    load_all, apply_theme, section_header,
    GREEN, GREEN_PALE, RED, RED_PALE, AMBER, AMBER_PALE,
    GRAY, BLUE, BLUE_PALE, SEG_COLORS, SEG_LABELS, AXIS_BASE
)
from cate_engine import (
    compute_cate, compute_variable_importance,
    FEATURE_LABELS, FEATURE_INTERPRETATION, FEATURES
)

FEATURE_LABELS = {
    "bet_freq_30d":           "Bet frequency (last 30 days)",
    "days_since_last_bet":    "Days since last bet",
    "days_since_last_deposit":"Days since last deposit",
    "ngr_90d":                "NGR (last 90 days)",
    "propensity_score":       "Organic bet probability",
    "parlay_rate":            "Parlay rate",
    "promo_redemption_rate":  "Past promo redemption rate",
}

FEATURE_INTERPRETATION = {
    "bet_freq_30d": (
        "Less frequent bettors respond more to promos — they have more room to increase activity. "
        "Highly active users are already at their ceiling."
    ),
    "days_since_last_bet": (
        "Users who have been inactive longer show higher causal lift — "
        "a promo is the nudge that brings them back. But very long inactivity (>60 days) "
        "may indicate permanent churn where even promos don't help."
    ),
    "days_since_last_deposit": (
        "Users who haven't deposited recently are more responsive to deposit bonuses specifically — "
        "the offer creates a concrete reason to fund their account."
    ),
    "ngr_90d": (
        "Higher-value users tend to show larger absolute NGR lifts, but the causal mechanism "
        "is different — they bet more per session when nudged, not just more often."
    ),
    "propensity_score": (
        "Lower organic bet probability strongly predicts higher causal lift — "
        "these users need the promo to engage. High-propensity users show near-zero CATE "
        "because they'd bet anyway."
    ),
    "parlay_rate": (
        "High parlay users show lower causal lift from deposit bonuses but higher lift "
        "from parlay-specific promotions. This is a promo-type matching signal, not just a targeting signal."
    ),
    "promo_redemption_rate": (
        "Users who have redeemed past promos show higher causal lift — "
        "they're promo-responsive by behavioral history. But very high redemption rates "
        "with low NGR contribution may indicate bonus seekers."
    ),
}

@st.cache_data
def get_cate_and_importance():
    users, *_ = load_all()
    cate_df  = compute_cate()
    vi_df, _ = compute_variable_importance()
    return cate_df, vi_df

def render():
    users, campaigns, exposures, daily = load_all()

    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <h1 style="font-size:22px;font-weight:600;color:#1A252F;margin:0;">
            CATE drivers</h1>
        <p style="font-size:14px;color:{GRAY};margin:4px 0 0;">
            What user characteristics drive treatment effect heterogeneity —
            who responds more to promos, and why.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#F8F9FA;border-radius:10px;padding:14px 18px;margin-bottom:24px;
                border-left:3px solid {BLUE};">
        <div style="font-size:13px;color:#2C3E50;line-height:1.65;">
            Variable importance is derived natively from a Causal Forest
            (econml.grf.CausalForest, n=500 trees) fit on each campaign's
            holdout experiment data. The forest splits on treatment effect
            heterogeneity — not prediction accuracy — so importance scores
            reflect which features drive <em>who responds causally</em> to a promo,
            not just who bets. Pooled across all 4 campaigns weighted by sample size.
            n = 1,000 users · Jan – Dec 2025.
        </div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Fitting causal forests across all campaigns..."):
        cate_df, vi_df = get_cate_and_importance()

    if vi_df.empty:
        st.warning("Not enough data to compute variable importance.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # VARIABLE IMPORTANCE — main chart
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "Variable importance — causal forest",
        "How much each feature drives treatment effect heterogeneity across the forest's splits. "
        "Direction shows whether higher values predict more (+) or less (−) causal lift."
    )

    vi_sorted = vi_df.sort_values("importance", ascending=True).copy()
    vi_sorted["bar_color"] = vi_sorted["direction"].map({"+": GREEN, "-": RED})
    vi_sorted["imp_pct"]   = vi_sorted["importance"] * 100

    fig_vi = go.Figure()
    fig_vi.add_trace(go.Bar(
        x=vi_sorted["imp_pct"],
        y=vi_sorted["label"],
        orientation="h",
        marker_color=vi_sorted["bar_color"],
        text=[f"{v:.1f}%" for v in vi_sorted["imp_pct"]],
        textposition="outside",
        textfont_size=11,
    ))

    apply_theme(fig_vi, height=340)
    fig_vi.update_layout(
        xaxis_title="Variable importance (% of total treatment effect splits)",
        yaxis_title="",
        showlegend=False,
        xaxis=dict(**AXIS_BASE, ticksuffix="%"),
    )

    # Legend for direction
    fig_vi.add_annotation(
        x=vi_sorted["imp_pct"].max() * 0.95, y=0.5,
        text="Green = higher value → more causal lift<br>Red = lower value → more causal lift",
        showarrow=False,
        font=dict(size=10, color=GRAY),
        align="right",
        xanchor="right",
    )

    st.plotly_chart(fig_vi, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TOP FEATURE DEEP-DIVES — CATE by quartile
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "Causal lift by feature quartile",
        "For each top driver, how does mean CATE vary across the distribution of that feature? "
        "Shows non-linear effects and where the sharpest thresholds lie."
    )

    top4 = vi_df.head(4)
    feat_cols_used = [f for f in FEATURES if f in users.columns]

    # Build quartile CATE from actual data
    reliable = cate_df[cate_df.reliable].copy()
    merged   = reliable.merge(users[["user_id"] + feat_cols_used], on="user_id", how="left")

    cols = st.columns(2)
    for idx, (_, row) in enumerate(top4.iterrows()):
        feat  = row["feature"]
        label = row["label"]
        interp = row.get("interpretation", "")

        if feat not in merged.columns:
            continue

        col = cols[idx % 2]
        with col:
            try:
                merged["_q"] = pd.qcut(merged[feat], q=4, labels=False, duplicates="drop")
                q_means  = merged.groupby("_q")["cate"].mean().values
                q_counts = merged.groupby("_q")["cate"].count().values
                q_labels = ["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"][:len(q_means)]
            except Exception:
                continue

            q_colors = []
            for v in q_means:
                if v >= 1.5:   q_colors.append(GREEN)
                elif v >= 0.3: q_colors.append("#5DADE2")
                elif v >= 0:   q_colors.append(AMBER)
                else:          q_colors.append(RED)

            fig_q = go.Figure(go.Bar(
                x=q_labels,
                y=q_means,
                marker_color=q_colors,
                text=[f"${v:.2f}\n(n={c})" for v, c in zip(q_means, q_counts)],
                textposition="outside",
                textfont_size=10,
            ))
            fig_q.add_hline(y=0, line_color=GRAY, line_width=1)
            apply_theme(fig_q, title=label, height=250)
            fig_q.update_layout(
                xaxis_title=f"{label} quartile",
                yaxis_title="Mean CATE ($/day)",
                yaxis=dict(**AXIS_BASE, tickprefix="$"),
                xaxis=dict(**AXIS_BASE),
                showlegend=False,
                margin=dict(l=0, r=20, t=44, b=0),
            )
            st.plotly_chart(fig_q, use_container_width=True)

            if interp:
                st.markdown(
                    f'<div style="font-size:12px;color:{GRAY};line-height:1.55;'
                    f'margin-top:-8px;margin-bottom:16px;">{interp}</div>',
                    unsafe_allow_html=True
                )

    # ══════════════════════════════════════════════════════════════════════════
    # CATE DISTRIBUTION BY SEGMENT
    # ══════════════════════════════════════════════════════════════════════════
    section_header(
        "CATE distribution by segment",
        "Individual causal lift estimates within each segment. "
        "Wide boxes = high within-segment heterogeneity — "
        "meaning segment-level targeting leaves money on the table."
    )

    fig_box = go.Figure()
    for seg in SEG_LABELS:
        grp = cate_df[cate_df.segment_label == seg]
        if grp.empty:
            continue
        fig_box.add_trace(go.Box(
            y=grp["cate"].values,
            name=SEG_LABELS[seg],
            marker_color=SEG_COLORS.get(seg, GRAY),
            boxmean=True,
            line_width=1.5,
        ))

    fig_box.add_hline(y=0, line_color=GRAY, line_width=1, line_dash="dot",
                      annotation_text="Zero lift",
                      annotation_font_size=10)
    apply_theme(fig_box, height=300)
    fig_box.update_layout(
        yaxis_title="Individual CATE ($/user/day)",
        xaxis_title="",
        yaxis=dict(**AXIS_BASE, tickprefix="$"),
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown(
        f'<div style="font-size:12px;color:{GRAY};line-height:1.6;">'
        f'Cross (+) shows mean CATE per segment. '
        f'Users within the same segment can have very different causal responses — '
        f'the Campaign Planner ranks by individual CATE to capture this variation '
        f'that segment rules cannot.</div>',
        unsafe_allow_html=True
    )

