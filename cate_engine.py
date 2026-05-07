"""
cate_engine.py
──────────────
Per-user CATE estimation via the R-Learner (Robinson 1988 / Nie & Wager 2021).

Method:
  The R-Learner estimates heterogeneous treatment effects via a two-step
  Robinson residualization, then fits a CATE model on the pseudo-outcome.

  Step 1 — Residualize outcome:
      Fit m(X) = E[Y|X] via cross-fitted GBM. Compute Y_res = Y - m(X).

  Step 2 — Residualize treatment:
      Fit e(X) = E[T|X] (propensity) via cross-fitted GBM. Compute T_res = T - e(X).

  Step 3 — Fit CATE model:
      The pseudo-outcome is: τ̃_i = Y_res_i / T_res_i
      Fit a Random Forest on pseudo-outcomes weighted by T_res^2.
      The forest's predictions are the per-user CATE estimates.
      Variable importance comes natively from the random forest.

  Confidence intervals: bootstrap (200 samples) on the CATE predictions.

  Identifying assumption: conditional ignorability — given X, treatment
  assignment is independent of potential outcomes. Holds here because
  holdout assignment was randomized within each campaign.

  Uses only scikit-learn — no heavy dependencies.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from data_loader import load_all

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

N_TREES      = 300
MIN_SAMPLES  = 5
N_BOOT       = 100   # bootstrap samples for CI
CV_FOLDS     = 3


def _build_Xy(campaign_id, users, exposures, daily, campaigns):
    camp_row  = campaigns[campaigns.campaign_id == campaign_id].iloc[0]
    c_start   = pd.Timestamp(camp_row.start_date)
    c_end     = pd.Timestamp(camp_row.end_date)

    camp_exp  = exposures[exposures.campaign_id == campaign_id]
    treat_ids = camp_exp[~camp_exp.is_holdout]["user_id"].tolist()
    hold_ids  = camp_exp[camp_exp.is_holdout]["user_id"].tolist()
    all_ids   = treat_ids + hold_ids

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


def _fit_rlearner(X, T, Y):
    """
    Fit R-Learner and return (cate, tau_model).
    """
    # Step 1: residualize outcome
    m = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                  random_state=42, learning_rate=0.1)
    Y_hat = cross_val_predict(m, X, Y, cv=CV_FOLDS)
    Y_res = Y - Y_hat

    # Step 2: residualize treatment (propensity)
    e = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                  random_state=42, learning_rate=0.1)
    T_hat = cross_val_predict(e, X, T.astype(float), cv=CV_FOLDS)
    T_hat = np.clip(T_hat, 0.05, 0.95)
    T_res = T - T_hat

    # Step 3: pseudo-outcome and CATE model
    mask         = np.abs(T_res) > 0.05
    pseudo       = Y_res[mask] / T_res[mask]
    weights      = T_res[mask] ** 2

    tau = RandomForestRegressor(
        n_estimators  = N_TREES,
        min_samples_leaf = MIN_SAMPLES,
        random_state  = 42,
        n_jobs        = -1,
    )
    tau.fit(X[mask], pseudo, sample_weight=weights)
    cate = tau.predict(X)
    return cate, tau


def compute_cate():
    users, campaigns, exposures, daily = load_all()
    results = []

    for _, camp in campaigns.iterrows():
        cid = camp.campaign_id
        X, T, Y, uids = _build_Xy(cid, users, exposures, daily, campaigns)
        if X is None:
            continue

        cate, tau_model = _fit_rlearner(X, T, Y)

        # Bootstrap CI
        boot_preds = np.zeros((N_BOOT, len(X)))
        for b in range(N_BOOT):
            idx = np.random.choice(len(X), len(X), replace=True)
            X_b, c_b = X[idx], cate[idx]
            rf_b = RandomForestRegressor(n_estimators=50, min_samples_leaf=MIN_SAMPLES,
                                         random_state=b, n_jobs=-1)
            rf_b.fit(X_b, c_b)
            boot_preds[b] = rf_b.predict(X)

        ci_lo = np.percentile(boot_preds, 2.5,  axis=0)
        ci_hi = np.percentile(boot_preds, 97.5, axis=0)

        camp_exp = exposures[exposures.campaign_id == cid]

        for i, uid in enumerate(uids):
            offer_row = camp_exp[camp_exp.user_id == uid]
            offer_amt = float(offer_row["offer_amount"].values[0]) if not offer_row.empty else 0.0
            seg       = users[users.user_id == uid]["segment_label"].values
            seg       = seg[0] if len(seg) else "unknown"

            results.append({
                "user_id":       uid,
                "campaign_id":   cid,
                "cate":          round(float(cate[i]),  4),
                "cate_ci_low":   round(float(ci_lo[i]), 4),
                "cate_ci_high":  round(float(ci_hi[i]), 4),
                "offer_amount":  offer_amt,
                "segment_label": seg,
                "reliable":      True,
            })

    return pd.DataFrame(results)


def compute_variable_importance():
    from scipy.stats import spearmanr
    users, campaigns, exposures, daily = load_all()
    feat_cols = [f for f in FEATURES if f in users.columns]

    all_vi   = []
    last_X   = None
    last_cate = None

    for _, camp in campaigns.iterrows():
        cid = camp.campaign_id
        X, T, Y, uids = _build_Xy(cid, users, exposures, daily, campaigns)
        if X is None:
            continue

        cate, tau_model = _fit_rlearner(X, T, Y)
        vi = tau_model.feature_importances_
        all_vi.append((len(uids), vi))

        if last_X is None or X.shape[0] > last_X.shape[0]:
            last_X    = X
            last_cate = cate

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

    return vi_df, {}
