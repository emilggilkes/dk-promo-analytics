import streamlit as st

st.set_page_config(
    page_title  = "DraftKings · Promo Analytics",
    page_icon   = "🎯",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0F2318;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #C8D8CC !important; }
[data-testid="stSidebar"] .stRadio label { 
    color: #C8D8CC !important; 
    font-size: 14px;
}
[data-testid="stSidebar"] hr { border-color: #1E3A2A; }

/* Radio nav pills */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
    padding: 6px 0;
}

/* Main area */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Dividers */
hr { border-color: #EAECEE; margin: 1.5rem 0; }

/* Selectbox / slider labels */
.stSelectbox label, .stSlider label, .stMultiSelect label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #5D6D7E !important;
}

/* Streamlit metric overrides */
[data-testid="metric-container"] {
    background: #fff;
    border: 1px solid #E5E8EA;
    border-radius: 10px;
    padding: 14px 16px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #F4F6F7;
    border-radius: 8px;
    padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 16px;
    color: #7F8C8D;
}
.stTabs [aria-selected="true"] {
    background: #fff !important;
    color: #1B5E3B !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #E5E8EA; border-radius: 8px; }

/* Expander */
.streamlit-expanderHeader { font-size: 13px; font-weight: 500; color: #2C3E50; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 24px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="background:#1B5E3B;border-radius:8px;width:32px;height:32px;
                        display:flex;align-items:center;justify-content:center;
                        font-weight:700;font-size:13px;color:#fff;">DK</div>
            <div>
                <div style="font-size:15px;font-weight:600;color:#fff;">Promo Analytics</div>
                <div style="font-size:11px;color:#7A9E85;">Powered by Sigma²</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    view = st.radio(
        "View",
        options=["Campaign autopsy", "Campaign planner", "Score drivers"],
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#4A6E56;line-height:1.6;">
        <div style="margin-bottom:8px;font-weight:600;color:#7A9E85;
                    text-transform:uppercase;letter-spacing:0.05em;font-size:10px;">
            Data</div>
        100 users · 4 campaigns<br/>
        Jan – Dec 2025<br/>
        36,500 daily observations
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#4A6E56;line-height:1.8;">
        <div style="margin-bottom:6px;font-weight:600;color:#7A9E85;
                    text-transform:uppercase;letter-spacing:0.05em;font-size:10px;">
            Methods</div>
        Synthetic control<br/>
        HTE / CATE estimation<br/>
        Cohort decay analysis<br/>
        Dose-response modeling<br/>
        Constrained optimization<br/>
        Regression-based audit
    </div>
    """, unsafe_allow_html=True)

# ── ROUTE VIEWS ───────────────────────────────────────────────────────────────
if view == "Campaign autopsy":
    from retrospective import render
    render()
elif view == "Campaign planner":
    from prospective import render
    render()
else:
    from score_drivers import render
    render()
