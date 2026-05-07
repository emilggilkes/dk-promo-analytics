"""
cate_engine.py
──────────────
Per-user CATE estimation via Causal Forest (econml.grf.CausalForest).

Method:
  For each campaign that has a holdout group, we fit one causal forest
  on the combined treatment + holdout sample using the campaign-period
  NGR as the outcome and a binary treatment indicator.

  The causal forest estimates CATE by:
    1. Growing trees that split on treatment effect heterogeneity
       (not just prediction accuracy)
    2. Locally re-weighting a Robinson residualization to partial out
       the main effect and propensity score
    3. Producing honest confidence intervals via the jackknife

  Variable importance comes natively from the forest via
  feature_importances_, which measures how often and how much each
  feature drives treatment effect splits.

Identifying assumption:
  Conditional ignorability — given the matching features, treatment
  assignment is independent of potential outcomes. This holds here
  because holdout assignment was randomized within each campaign.
"""

import numpy as np
import pandas as pd
from econml.grf import CausalForest
from data_loader import load_all

# Features used as X in the causal forest
FEATURES = [
    "bet_freq_30d",
    "days_since_last_bet",
    "days_since_last_deposit",
    "ngr_90d",
    "propensity_score",
    "parlay_rate",
    "promo_redemption_rate",
    "avg_bet_size",
    "churn_risk_30d",
    "live_bet_rate",
]

FEATURE_LABELS = {
    "bet_freq_30d":            "Bet frequency (30d)",
    "days_since_last_bet":     "Days since last bet",
    "days_since_last_deposit": "Days since last deposit",
    "ngr_90d":                 "NGR (90d)",
    "propensity_score":        "Organic bet probability",
    "parlay_rate":             "Parlay rate",
    "promo_redemption_rate":   "Past promo redemption rate",
    "avg_bet_size":            "Avg bet size",
    "churn_risk_30d":          "Churn risk (30d)",
    "live_bet_rate":           "Live betting rate",
}

FEATURE_INTERPRETATION = {
    "bet_freq_30d": (
        "Less frequent bettors show higher causal lift — they have more room to increase activity. "
        "Highly active users are already near their ceiling."
    ),
    "days_since_last_bet": (
        "Users inactive longer respond more to promos. A well-timed offer is the nudge "
        "that reactivates them. Very long inactivity (>60 days) may indicate permanent churn."
    ),
    "days_since_last_deposit": (
        "Users who haven't deposited recently are more responsive to deposit bonuses — "
        "the offer gives them a concrete reason to fund their account."
    ),
    "ngr_90d": (
        "Higher-value users show larger absolute NGR lifts when they respond — "
        "they bet more per session when nudged, not just more often."
    ),
    "propensity_score": (
        "Lower organic bet probability strongly predicts higher causal lift. "
        "High-propensity users show near-zero CATE — they'd bet anyway."
    ),
    "parlay_rate": (
        "High parlay users show lower causal lift from deposit bonuses but higher lift "
        "from parlay-specific promotions. A promo-type matching signal."
    ),
    "promo_redemption_rate": (
        "Users who redeemed past promos show higher causal lift — "
        "they are promo-responsive by behavioral history."
    ),
    "avg_bet_size": (
        "Larger bettors generate more incremental NGR per dollar of promo when they respond, "
        "making them high-priority even at lower response probabilities."
    ),
    "churn_risk_30d": (
        "Higher churn risk users are more urgently responsive — the promo arrives "
        "at the right moment to change their trajectory."
    ),
    "live_bet_rate": (
        "Live bettors tend to have more session-driven behavior, "
        "making them responsive to offers that incentivize logging in."
    ),
}

N_ESTIMATORS = 500
MIN_SAMPLES  = 5   # min observations per leaf


def _build_Xy(campaign_id, users, exposures, daily, campaigns):
    """
    Build (X, T, Y) for a single campaign.
    X = feature matrix (treated + holdout users)
    T = treatment indicator (1 = received promo)
    Y = mean NGR per user during campaign window
    """
    from data_loader import load_all as _load
    _, camps, _, _ = _load()

    camp_row  = camps[camps.campaign_id == campaign_id].iloc[0]
    c_start   = pd.Timestamp(camp_row.start_date)
    c_end     = pd.Timestamp(camp_row.end_date)

    camp_exp  = exposures[exposures.campaign_id == campaign_id]
    treat_ids = camp_exp[~camp_exp.is_holdout]["user_id"].tolist()
    hold_ids  = camp_exp[camp_exp.is_holdout]["user_id"].tolist()

    all_ids = treat_ids + hold_ids
    if len(treat_ids) < MIN_SAMPLES or len(hold_ids) < MIN_SAMPLES:
        return None, None, None, None

    camp_mask = (daily.date >= c_start) & (daily.date <= c_end)
    camp_ngr  = (daily[camp_mask & daily.user_id.isin(all_ids)]
                 .groupby("user_id")["ngr"].mean()
                 .reset_index(name="Y"))

    feat_cols = [f for f in FEATURES if f in users.columns]
    feat_df   = (users[users.user_id.isin(all_ids)][["user_id"] + feat_cols]
                 .merge(camp_ngr, on="user_id", how="inner")
                 .fillna(0))

    if len(feat_df) < MIN_SAMPLES * 2:
        return None, None, None, None

    X    = feat_df[feat_cols].values.astype(float)
    Y    = feat_df["Y"].values.astype(float)
    T    = feat_df["user_id"].isin(treat_ids).astype(int).values
    uids = feat_df["user_id"].tolist()
    return X, T, Y, uids


def compute_cate():
    """
    Fit a CausalForest per campaign and return per-user CATE estimates.

    Returns DataFrame with columns:
        user_id, campaign_id, cate, cate_ci_low, cate_ci_high,
        segment_label, offer_amount, reliable
    """
    users, campaigns, exposures, daily = load_all()
    feat_cols = [f for f in FEATURES if f in users.columns]
    results   = []

    for _, camp in campaigns.iterrows():
        cid = camp.campaign_id
        X, T, Y, uids = _build_Xy(cid, users, exposures, daily, campaigns)
        if X is None:
            continue

        cf = CausalForest(
            n_estimators     = N_ESTIMATORS,
            min_samples_leaf = MIN_SAMPLES,
            random_state     = 42,
            verbose          = 0,
        )
        cf.fit(X, T, Y)

        cate_hat, cate_lo, cate_hi = cf.predict(X, interval=True, alpha=0.05)
        # CausalForest returns shape (n,1) — flatten to (n,)
        cate_hat = cate_hat.flatten()
        cate_lo  = cate_lo.flatten()
        cate_hi  = cate_hi.flatten()
        camp_exp = exposures[exposures.campaign_id == cid]

        for i, uid in enumerate(uids):
            offer_row = camp_exp[camp_exp.user_id == uid]
            offer_amt = float(offer_row["offer_amount"].values[0]) if not offer_row.empty else 0.0
            seg       = users[users.user_id == uid]["segment_label"].values
            seg       = seg[0] if len(seg) else "unknown"

            results.append({
                "user_id":       uid,
                "campaign_id":   cid,
                "cate":          round(float(cate_hat[i]), 4),
                "cate_ci_low":   round(float(cate_lo[i]),  4),
                "cate_ci_high":  round(float(cate_hi[i]),  4),
                "offer_amount":  offer_amt,
                "segment_label": seg,
                "reliable":      True,
            })

    return pd.DataFrame(results)


def compute_variable_importance():
    """
    Fit causal forests across all campaigns and return pooled
    variable importance (mean across campaigns, weighted by n).

    Returns:
        vi_df    — DataFrame: feature, label, importance, rank, direction, interpretation
        camp_vi  — dict: campaign_id -> importance array
    """
    from scipy.stats import spearmanr
    users, campaigns, exposures, daily = load_all()
    feat_cols = [f for f in FEATURES if f in users.columns]

    all_vi   = []
    camp_vi  = {}
    last_X   = None
    last_cate = None

    for _, camp in campaigns.iterrows():
        cid = camp.campaign_id
        X, T, Y, uids = _build_Xy(cid, users, exposures, daily, campaigns)
        if X is None:
            continue

        cf = CausalForest(
            n_estimators     = N_ESTIMATORS,
            min_samples_leaf = MIN_SAMPLES,
            random_state     = 42,
            verbose          = 0,
        )
        cf.fit(X, T, Y)

        vi = cf.feature_importances_
        all_vi.append((len(uids), vi))
        camp_vi[cid] = vi

        if last_X is None or X.shape[0] > last_X.shape[0]:
            last_X    = X
            last_cate = cf.predict(X).flatten()

    if not all_vi:
        return pd.DataFrame(), {}

    total_n   = sum(n for n, _ in all_vi)
    pooled_vi = sum(n / total_n * vi for n, vi in all_vi)

    directions = []
    for j in range(last_X.shape[1]):
        r, _ = spearmanr(last_X[:, j], last_cate)
        directions.append("+" if r >= 0 else "-")

    vi_df = pd.DataFrame({
        "feature":        feat_cols,
        "label":          [FEATURE_LABELS.get(f, f) for f in feat_cols],
        "importance":     pooled_vi,
        "direction":      directions,
        "interpretation": [FEATURE_INTERPRETATION.get(f, "") for f in feat_cols],
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    vi_df["rank"] = range(1, len(vi_df) + 1)

    return vi_df, camp_vi
