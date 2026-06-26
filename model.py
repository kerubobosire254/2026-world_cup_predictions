

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import joblib
import xgboost as xgb
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

from data_prep import (
    load_and_prepare,
    build_poisson_rows,
    XGBOOST_FEATURES,
    POISSON_FEATURES,
)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(train, test, le):
    X_train = train[XGBOOST_FEATURES]
    y_train = train['outcome_encoded']
    X_test  = test[XGBOOST_FEATURES]
    y_test  = test['outcome_encoded']

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=3,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"  Cross-val accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    train_marginals = np.bincount(y_train.values) / len(y_train)
    raw_probas  = model.predict_proba(X_test)
    model_marg  = raw_probas.mean(axis=0)
    corrected   = raw_probas * train_marginals / model_marg
    corrected  /= corrected.sum(axis=1, keepdims=True)
    preds       = np.argmax(corrected, axis=1)

    acc = accuracy_score(y_test, preds)
    print(f"  Test accuracy (2022 WC):  {acc:.1%}")
    print(classification_report(y_test, preds, target_names=le.classes_))

    return model, train_marginals, acc, cv_scores.mean()


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN POISSON
# ─────────────────────────────────────────────────────────────────────────────

def train_poisson(train, test):
    p_train = build_poisson_rows(train)
    p_test  = build_poisson_rows(test)

    train_h = p_train[p_train['is_home'] == 1]
    train_a = p_train[p_train['is_home'] == 0]
    test_h  = p_test[p_test['is_home'] == 1]
    test_a  = p_test[p_test['is_home'] == 0]

    home_model = PoissonRegressor(alpha=0.1, max_iter=3000)
    home_model.fit(train_h[POISSON_FEATURES], train_h['goals'])

    away_model = PoissonRegressor(alpha=0.1, max_iter=3000)
    away_model.fit(train_a[POISSON_FEATURES], train_a['goals'])

    home_mae = mean_absolute_error(test_h['goals'], home_model.predict(test_h[POISSON_FEATURES]))
    away_mae = mean_absolute_error(test_a['goals'], away_model.predict(test_a[POISSON_FEATURES]))
    combined = (home_mae + away_mae) / 2

    print(f"  Home goals MAE: {home_mae:.3f}")
    print(f"  Away goals MAE: {away_mae:.3f}")
    print(f"  Combined MAE:   {combined:.3f}")

    return home_model, away_model, combined


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN EVERYTHING + SAVE
# ─────────────────────────────────────────────────────────────────────────────

def train_all(
    data_path='data/master_historical_features (1).csv',
    save_path='models.joblib',
):
    """
    Full training pipeline. Saves to models.joblib using joblib — NOT pickle.
    joblib handles sklearn/numpy objects cleanly across Python versions.
    """
    print("=" * 55)
    print("  WORLD CUP 2026 — TRAINING MODELS")
    print("=" * 55)

    print("\n[1/3] Loading + preparing historical data...")
    df, train, test, le = load_and_prepare(data_path)
    print(f"  Training: {len(train)} matches (1930–2018)")
    print(f"  Testing:  {len(test)} matches (2022 WC)")

    print("\n[2/3] XGBoost — Win / Draw / Loss classifier...")
    xgb_model, train_marginals, xgb_acc, xgb_cv = train_xgboost(train, test, le)

    print("\n[3/3] Poisson — Scoreline predictor...")
    home_poisson, away_poisson, poisson_mae = train_poisson(train, test)

    bundle = {
        'xgb_model':        xgb_model,
        'home_poisson':     home_poisson,
        'away_poisson':     away_poisson,
        'train_marginals':  train_marginals,
        'label_encoder':    le,
        'xgb_features':     XGBOOST_FEATURES,
        'poisson_features': POISSON_FEATURES,
        'xgb_accuracy':     xgb_acc,
        'xgb_cv_accuracy':  xgb_cv,
        'poisson_mae':      poisson_mae,
    }

    joblib.dump(bundle, save_path)

    print(f"\n✅  Models saved → {save_path}")
    print(f"    XGBoost test accuracy:  {xgb_acc:.1%}")
    print(f"    XGBoost CV accuracy:    {xgb_cv:.1%}")
    print(f"    Poisson combined MAE:   {poisson_mae:.3f} goals")
    print("\nRun: streamlit run app.py")

    return bundle


if __name__ == '__main__':
    train_all()