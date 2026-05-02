import streamlit as st
import plotly.graph_objects as go
import numpy as np
from data_loader import (
    apply_theme, section_header,
    GREEN, GREEN_PALE, RED, RED_PALE, AMBER, AMBER_PALE,
    GRAY, BLUE, BLUE_PALE, AXIS_BASE
)

# ── SCORE DEFINITIONS ─────────────────────────────────────────────────────────
# Each score has: description, color palette, and a list of feature drivers.
# Impact values are illustrative — representative of what a trained model
# on DraftKings data would surface.

SCORES = {
    "Propensity Score": {
        "tagline": "How likely is this user to bet without any promo?",
        "description": (
            "Measures a user's baseline betting tendency from their organic behavior alone. "
            "A high score means they're already highly active — sending them a promo is wasted spend. "
            "A low score means they need a reason to engage."
        ),
        "color": BLUE,
        "bg":    BLUE_PALE,
        "interpretation": {
            "high": "Always-on bettor — do not target",
            "mid":  "Situational bettor — target selectively",
            "low":  "Dormant / lapsed — prime promo candidate",
        },
        "features": [
            {"name": "Bet frequency (last 90 days)",              "impact": 34, "direction": "+"},
            {"name": "Days since last bet",                        "impact": 28, "direction": "-"},
            {"name": "Number of active betting weeks",             "impact": 18, "direction": "+"},
            {"name": "Avg daily sessions",                         "impact": 11, "direction": "+"},
            {"name": "Live betting rate",                          "impact":  9, "direction": "+"},
            {"name": "Deposit frequency (last 60 days)",           "impact":  8, "direction": "+"},
            {"name": "Number of sports wagered on",                "impact":  7, "direction": "+"},
            {"name": "Avg bet size",                               "impact":  6, "direction": "+"},
            {"name": "Weekend vs weekday bet ratio",               "impact":  5, "direction": "+"},
            {"name": "Account age (days)",                         "impact":  4, "direction": "+"},
        ],
    },
    "Lift Probability": {
        "tagline": "How likely is this user to increase activity if given a promo?",
        "description": (
            "Estimates the probability that a promo will causally change this user's behavior — "
            "not just correlate with it. Trained on historical holdout experiments. "
            "High-propensity users rarely appear here because they'd bet anyway."
        ),
        "color": GREEN,
        "bg":    GREEN_PALE,
        "interpretation": {
            "high": "Strong responder — high ROI target",
            "mid":  "Moderate responder — worth including",
            "low":  "Unlikely to respond — lower priority",
        },
        "features": [
            {"name": "Past promo redemption rate",                 "impact": 31, "direction": "+"},
            {"name": "Bet frequency decline (last 30d)",           "impact": 24, "direction": "+"},
            {"name": "Deposit amount on promo days",               "impact": 19, "direction": "+"},
            {"name": "Days between promo send and first redeem",   "impact": 14, "direction": "-"},
            {"name": "Parlay rate",                                "impact": 12, "direction": "+"},
            {"name": "NGR change after last promo",                "impact": 10, "direction": "+"},
            {"name": "Promo offer size sensitivity",               "impact":  9, "direction": "+"},
            {"name": "Days since last deposit",                    "impact":  8, "direction": "+"},
            {"name": "Number of bet types used",                   "impact":  6, "direction": "+"},
            {"name": "Time between promos received",               "impact":  5, "direction": "-"},
        ],
    },
    "Churn Risk": {
        "tagline": "How likely is this user to go dormant in the next 30 days?",
        "description": (
            "Predicts the probability of zero bets in the next 30 days based on "
            "recent behavioral signals. High churn risk users are urgent — "
            "they're on the way out and a well-timed promo is the cheapest retention tool."
        ),
        "color": "#C0392B",
        "bg":    RED_PALE,
        "interpretation": {
            "high": "About to lapse — act now",
            "mid":  "Showing early warning signs",
            "low":  "Stable — lower urgency",
        },
        "features": [
            {"name": "Days since last bet",                        "impact": 36, "direction": "+"},
            {"name": "Consecutive losing streak",                  "impact": 22, "direction": "+"},
            {"name": "Bet frequency trend (30d slope)",            "impact": 18, "direction": "-"},
            {"name": "Avg session length decline",                 "impact": 13, "direction": "+"},
            {"name": "Sport season in/out of calendar",            "impact": 11, "direction": "+"},
            {"name": "Win rate over last 20 bets",                 "impact":  9, "direction": "-"},
            {"name": "Deposit recency",                            "impact":  8, "direction": "-"},
            {"name": "Number of days with zero activity (30d)",    "impact":  7, "direction": "+"},
            {"name": "Reduction in avg bet size",                  "impact":  6, "direction": "+"},
            {"name": "Previously churned and reactivated",         "impact":  5, "direction": "+"},
        ],
    },
}

def impact_color(impact, color):
    """Return darker shade for higher impact bars."""
    if impact >= 30: return color
    if impact >= 18: return color + "BB"
    return color + "77"

def direction_label(d):
    return "↑ increases score" if d == "+" else "↓ decreases score"

def render():
    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <h1 style="font-size:22px;font-weight:600;color:#1A252F;margin:0;">
            Score drivers</h1>
        <p style="font-size:14px;color:{GRAY};margin:4px 0 0;">
            What user behaviors drive each predictive score — based on 100 users over Jan – Dec 2025</p>
    </div>""", unsafe_allow_html=True)

    # ── SCORE SELECTOR ────────────────────────────────────────────────────────
    selected = st.tabs(list(SCORES.keys()))

    for tab, (score_name, score) in zip(selected, SCORES.items()):
        with tab:
            color   = score["color"]
            bg      = score["bg"]
            features = score["features"]

            # Header block
            st.markdown(f"""
            <div style="background:{bg};border-radius:10px;padding:16px 20px;margin:16px 0 24px;">
                <div style="font-size:16px;font-weight:600;color:{color};margin-bottom:6px;">
                    {score_name}</div>
                <div style="font-size:13px;color:#2C3E50;line-height:1.65;margin-bottom:12px;">
                    {score["description"]}</div>
                <div style="display:flex;gap:10px;flex-wrap:wrap;">
                    <div style="background:rgba(255,255,255,0.7);border-radius:6px;
                                padding:6px 12px;font-size:12px;color:{color};font-weight:500;">
                        ● High (&gt;0.7): {score["interpretation"]["high"]}</div>
                    <div style="background:rgba(255,255,255,0.7);border-radius:6px;
                                padding:6px 12px;font-size:12px;color:{AMBER};font-weight:500;">
                        ● Mid (0.4–0.7): {score["interpretation"]["mid"]}</div>
                    <div style="background:rgba(255,255,255,0.7);border-radius:6px;
                                padding:6px 12px;font-size:12px;color:{GRAY};font-weight:500;">
                        ● Low (&lt;0.4): {score["interpretation"]["low"]}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── FULL WIDTH: Feature importance bars ───────────────────────────
            section_header(
                "Key factors",
                "User behaviors ranked by predictive impact on this score"
            )

            # Split features into two columns for full-width layout
            mid = (len(features) + 1) // 2
            col_left, col_right = st.columns(2)

            for col, chunk in [(col_left, features[:mid]), (col_right, features[mid:])]:
                with col:
                    for feat in chunk:
                        pct   = feat["impact"]
                        fname = feat["name"]
                        direc = feat["direction"]
                        bar_color = color if pct >= 20 else (AMBER if pct >= 10 else GRAY)

                        if pct > 20:
                            band_bg, band_fg = bg, color
                        elif pct >= 10:
                            band_bg, band_fg = AMBER_PALE, AMBER
                        else:
                            band_bg, band_fg = "#F2F3F4", GRAY

                        st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:10px;
                                    padding:9px 0;border-bottom:0.5px solid #EAECEE;">
                            <div style="flex:1;font-size:13px;color:#2C3E50;">{fname}</div>
                            <div style="width:120px;height:6px;background:#EAECEE;
                                        border-radius:3px;overflow:hidden;flex-shrink:0;">
                                <div style="width:{min(pct*3, 120)}px;height:100%;
                                            background:{bar_color};border-radius:3px;"></div>
                            </div>
                            <div style="font-size:11px;font-weight:600;padding:2px 8px;
                                        border-radius:12px;background:{band_bg};color:{band_fg};
                                        min-width:38px;text-align:center;flex-shrink:0;">{pct}%</div>
                            <div style="font-size:10px;color:{GRAY};width:120px;flex-shrink:0;">
                                {direction_label(direc)}</div>
                        </div>""", unsafe_allow_html=True)

            st.markdown(
                f'<div style="font-size:11px;color:{GRAY};margin-top:12px;">'
                f'n = 100 users · Jan – Dec 2025 · Model updated nightly</div>',
                unsafe_allow_html=True
            )

            # ── SCORE DISTRIBUTION SIMULATION ────────────────────────────────
            section_header(
                "Score distribution across user base",
                "How the current user population is distributed — and what that means for targeting"
            )

            # Simulate a realistic score distribution per score type
            np.random.seed({"Propensity Score": 1, "Lift Probability": 2, "Churn Risk": 3}[score_name])

            if score_name == "Propensity Score":
                # Bimodal: many always-on + many dormant
                vals = np.concatenate([
                    np.random.beta(2, 6, 18000),   # low propensity (dormant)
                    np.random.beta(6, 2, 12000),   # high propensity (always-on)
                    np.random.beta(3, 3, 10000),   # mid
                ])
            elif score_name == "Lift Probability":
                # Right-skewed: most users have moderate-low lift prob
                vals = np.random.beta(2, 4, 40000)
            else:
                # Churn risk: most users are stable, long tail of high-risk
                vals = np.random.beta(1.5, 5, 40000)

            # Bin into 20 buckets
            counts, edges = np.histogram(vals, bins=20, range=(0, 1))
            bin_centers   = (edges[:-1] + edges[1:]) / 2

            # Color bars by zone
            bar_colors = []
            for c in bin_centers:
                if score_name == "Propensity Score":
                    bar_colors.append(RED if c > 0.65 else (AMBER if c > 0.35 else color))
                elif score_name == "Lift Probability":
                    bar_colors.append(color if c > 0.55 else (AMBER if c > 0.3 else GRAY))
                else:
                    bar_colors.append(RED if c > 0.6 else (AMBER if c > 0.35 else color))

            fig_dist = go.Figure(go.Bar(
                x=bin_centers,
                y=counts,
                marker_color=bar_colors,
                marker_line_width=0,
                width=0.045,
                hovertemplate="Score: %{x:.2f}<br>Users: %{y:,}<extra></extra>",
            ))

            # Threshold lines with labels
            if score_name == "Propensity Score":
                fig_dist.add_vline(x=0.65, line_color=RED, line_dash="dash", line_width=1,
                                   annotation_text="Always-on cutoff",
                                   annotation_font_size=10, annotation_font_color=RED)
                fig_dist.add_vline(x=0.35, line_color=AMBER, line_dash="dot", line_width=1,
                                   annotation_text="Mid threshold",
                                   annotation_font_size=10, annotation_font_color=AMBER)
            elif score_name == "Lift Probability":
                fig_dist.add_vline(x=0.55, line_color=color, line_dash="dash", line_width=1,
                                   annotation_text="High responder",
                                   annotation_font_size=10, annotation_font_color=color)
            else:
                fig_dist.add_vline(x=0.6, line_color=RED, line_dash="dash", line_width=1,
                                   annotation_text="High risk",
                                   annotation_font_size=10, annotation_font_color=RED)

            apply_theme(fig_dist, height=240)
            fig_dist.update_layout(
                xaxis_title=f"{score_name} (0 = low, 1 = high)",
                yaxis_title="Number of users",
                xaxis=dict(**AXIS_BASE, tickformat=".1f", range=[0, 1]),
                yaxis=dict(**AXIS_BASE, tickformat=","),
                bargap=0.05,
                showlegend=False,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Summary callout
            high_pct = (vals > 0.6).mean() * 100
            low_pct  = (vals < 0.35).mean() * 100

            if score_name == "Propensity Score":
                msg = (f"<b>{high_pct:.0f}%</b> of users are high-propensity (always-on) — "
                       f"they should be excluded from most promo campaigns to avoid budget waste. "
                       f"<b>{low_pct:.0f}%</b> are low-propensity and represent the highest-ROI targeting pool.")
                msg_color, msg_bg = BLUE, BLUE_PALE
            elif score_name == "Lift Probability":
                msg = (f"<b>{high_pct:.0f}%</b> of users show strong promo response probability — "
                       f"these are the users whose NGR is most likely to increase causally following an offer. "
                       f"Concentrating budget here maximises incremental return.")
                msg_color, msg_bg = GREEN, GREEN_PALE
            else:
                msg = (f"<b>{high_pct:.0f}%</b> of users are at high churn risk right now — "
                       f"a well-timed promo is the cheapest retention tool available. "
                       f"Waiting longer sharply reduces the probability of reactivation.")
                msg_color, msg_bg = RED, RED_PALE

            st.markdown(
                f'<div style="font-size:13px;color:{msg_color};background:{msg_bg};'
                f'padding:10px 14px;border-radius:8px;line-height:1.6;">{msg}</div>',
                unsafe_allow_html=True
            )
