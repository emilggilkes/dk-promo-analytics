import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

DATA = Path(__file__).parent / "data"

@st.cache_data
def load_users():
    df = pd.read_csv(DATA / "users.csv", parse_dates=["registration_date"])
    return df

@st.cache_data
def load_campaigns():
    df = pd.read_csv(DATA / "campaigns.csv", parse_dates=["start_date", "end_date"])
    return df

@st.cache_data
def load_exposures():
    df = pd.read_csv(DATA / "promo_exposures.csv", parse_dates=["exposure_date"])
    return df

@st.cache_data
def load_daily():
    df = pd.read_csv(DATA / "user_daily_metrics.csv", parse_dates=["date"])
    return df

def load_all():
    return load_users(), load_campaigns(), load_exposures(), load_daily()

# ── COLORS ────────────────────────────────────────────────────────────────────
GREEN       = "#1B5E3B"
GREEN_LIGHT = "#2E7D52"
GREEN_MID   = "#4CAF80"
GREEN_PALE  = "#E8F5EE"
RED         = "#C0392B"
RED_PALE    = "#FDECEA"
AMBER       = "#E67E22"
AMBER_PALE  = "#FEF5E7"
GRAY        = "#7F8C8D"
GRAY_LIGHT  = "#F4F6F7"
BLUE        = "#2471A3"
BLUE_PALE   = "#EBF5FB"

SEG_COLORS = {
    "dormant_high_value": GREEN,
    "casual":             BLUE,
    "bonus_seeker":       AMBER,
    "active_mid":         "#8E44AD",
    "always_on":          RED,
}

SEG_LABELS = {
    "dormant_high_value": "Dormant high-value",
    "casual":             "Casual",
    "bonus_seeker":       "Bonus seeker",
    "active_mid":         "Active mid-volume",
    "always_on":          "Always-on",
}

PROMO_COLORS = {
    "deposit_bonus": GREEN,
    "parlay_boost":  BLUE,
    "free_bet":      AMBER,
    "odds_boost":    "#8E44AD",
}

# ── PLOTLY BASE THEME ─────────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    font_family   = "Inter, sans-serif",
    font_color    = "#2C3E50",
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    margin        = dict(l=0, r=0, t=36, b=0),
    legend        = dict(
        bgcolor     = "rgba(255,255,255,0.9)",
        bordercolor = "#E5E8EA",
        borderwidth = 1,
        font_size   = 12,
    ),
)

AXIS_BASE = dict(
    showgrid      = True,
    gridcolor     = "#EAECEE",
    gridwidth     = 0.5,
    zeroline      = False,
    tickfont_size = 11,
    tickfont_color= "#7F8C8D",
    title_font_size = 12,
    title_font_color= "#5D6D7E",
)

def apply_theme(fig, title=None, height=None):
    layout = dict(**LAYOUT_BASE)
    if title:
        layout["title"] = dict(text=title, font_size=14, font_color="#1A252F", x=0, xanchor="left")
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(**AXIS_BASE)
    fig.update_yaxes(**AXIS_BASE)
    return fig

# ── METRIC CARD HTML ──────────────────────────────────────────────────────────
def metric_card(label, value, delta=None, delta_good=True, note=None):
    delta_html = ""
    if delta:
        color = GREEN if delta_good else RED
        arrow = "▲" if delta_good else "▼"
        delta_html = f'<div style="font-size:12px;color:{color};margin-top:2px;">{arrow} {delta}</div>'
    note_html = ""
    if note:
        note_html = f'<div style="font-size:11px;color:{GRAY};margin-top:4px;">{note}</div>'
    return f"""
    <div style="
        background:#fff;border:1px solid #E5E8EA;border-radius:10px;
        padding:14px 16px;height:100%;
    ">
        <div style="font-size:11px;color:{GRAY};text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:6px;">{label}</div>
        <div style="font-size:24px;font-weight:600;color:#1A252F;
                    font-family:'Inter',monospace;">{value}</div>
        {delta_html}{note_html}
    </div>"""

def confidence_badge(pct):
    if pct >= 85:
        bg, fg = GREEN_PALE, GREEN
    elif pct >= 70:
        bg, fg = AMBER_PALE, AMBER
    else:
        bg, fg = "#F2F3F4", GRAY
    return f'<span style="background:{bg};color:{fg};padding:2px 9px;border-radius:12px;font-size:11px;font-weight:600;">Confidence: {pct}%</span>'

def section_header(title, subtitle=None):
    sub = f'<p style="font-size:13px;color:{GRAY};margin:2px 0 0;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin:28px 0 16px;">
        <h3 style="font-size:15px;font-weight:600;color:#1A252F;margin:0;">{title}</h3>
        {sub}
    </div>""", unsafe_allow_html=True)
