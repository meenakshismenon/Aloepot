from __future__ import annotations

import json
import math
import pickle
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from preprocess import time_aware_split, validate_dataset

warnings.filterwarnings("ignore")

MODEL_PATH = Path("aloe_health_model.pkl")
REPORT_PATH = Path("model_training_report.json")
TARGET = "health_label"
LABELS = ["Happy", "Stressed", "Unhealthy"]
FEATURES = [
    "soil_moisture", "temperature", "humidity",
    "light_lux", "vpd", "last_watered_hours", "watering_event",
]


# ---------------------------------------------------------------------------
# Rule-based baseline (mirrors app.py fallback_health_rule)
# ---------------------------------------------------------------------------
def _rule_predict(soil, temp, humidity, light_lux, vpd) -> str:
    score = 100
    if soil <= 10:        score -= 28
    elif soil < 20:       score -= 12
    elif soil > 55:       score -= 20
    if temp < 18 or temp > 27: score -= 18
    if humidity > 60:     score -= 20
    light_pct = min(100, light_lux / 150.0)
    if light_pct < 35:    score -= 12
    if vpd > 1.8:         score -= 24
    if vpd < 0.4:         score -= 8
    score = max(0, min(100, score))
    if score >= 75: return "Happy"
    if score >= 45: return "Stressed"
    return "Unhealthy"


# ---------------------------------------------------------------------------
def train_and_evaluate(csv_path: str = "aloe_vera_indoor_master.csv") -> dict:
    print("=" * 62)
    print("  ALOE VERA ONLY — MODEL TRAINING & COMPARISON REPORT")
    print("=" * 62)

    df = validate_dataset(csv_path)

    print(f"\nAloe Vera dataset: {len(df)} rows (generic IoT pool excluded)")
    print("Data sources :", df["data_source"].value_counts().to_dict())
    print("Class balance:", df[TARGET].value_counts().to_dict())

    train_df, val_df, test_df = time_aware_split(df)
    print(f"\nSplit  =>  Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val,   y_val   = val_df[FEATURES],   val_df[TARGET]
    X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

    # ── Train three ML models ────────────────────────────────────────────────
    candidates = {
        "RandomForest":     RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=150, learning_rate=0.1),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    }

    results: dict = {}
    for name, model in candidates.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        val_preds = model.predict(X_val)
        results[name] = {
            "val_accuracy":       round(accuracy_score(y_val, val_preds), 4),
            "test_accuracy":      round(accuracy_score(y_test, preds), 4),
            "balanced_accuracy":  round(balanced_accuracy_score(y_test, preds), 4),
            "macro_f1":           round(f1_score(y_test, preds, average="macro"), 4),
            "confusion_matrix":   confusion_matrix(y_test, preds, labels=LABELS).tolist(),
            "classification_report": classification_report(y_test, preds, target_names=LABELS),
        }
        print(f"    Val Acc  : {results[name]['val_accuracy']:.4f}")
        print(f"    Test Acc : {results[name]['test_accuracy']:.4f}")
        print(f"    Bal Acc  : {results[name]['balanced_accuracy']:.4f}")
        print(f"    Macro F1 : {results[name]['macro_f1']:.4f}")

    # ── Rule-based baseline ──────────────────────────────────────────────────
    print("\n  Evaluating Rule-Based Baseline (mirrors app.py logic)...")
    rule_preds = [
        _rule_predict(r.soil_moisture, r.temperature, r.humidity, r.light_lux, r.vpd)
        for r in test_df.itertuples()
    ]
    rule_acc  = round(accuracy_score(y_test, rule_preds), 4)
    rule_bal  = round(balanced_accuracy_score(y_test, rule_preds), 4)
    rule_f1   = round(f1_score(y_test, rule_preds, average="macro"), 4)
    rule_cm   = confusion_matrix(y_test, rule_preds, labels=LABELS).tolist()
    rule_report = classification_report(y_test, rule_preds, target_names=LABELS)
    results["RuleBasedFallback"] = {
        "val_accuracy":       None,
        "test_accuracy":      rule_acc,
        "balanced_accuracy":  rule_bal,
        "macro_f1":           rule_f1,
        "confusion_matrix":   rule_cm,
        "classification_report": rule_report,
    }
    print(f"    Test Acc : {rule_acc:.4f}")
    print(f"    Bal Acc  : {rule_bal:.4f}")
    print(f"    Macro F1 : {rule_f1:.4f}")

    # ── Pick best ML model ───────────────────────────────────────────────────
    best_name = max(candidates, key=lambda k: results[k]["macro_f1"])
    gain_acc = round(results[best_name]["test_accuracy"] - rule_acc, 4)
    gain_f1  = round(results[best_name]["macro_f1"]      - rule_f1,  4)

    # Retrain best on train+val for production
    best_model = {
        "RandomForest":     RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=150, learning_rate=0.1),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    }[best_name]
    best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    with MODEL_PATH.open("wb") as fh:
        pickle.dump(best_model, fh)

    # ── Print full comparison table ──────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  COMPARISON SUMMARY  (Aloe Vera-only dataset)")
    print("=" * 62)
    print(f"{'Model':<22} {'Test Acc':>9} {'Bal Acc':>9} {'Macro F1':>9}")
    print("-" * 52)
    for name, r in results.items():
        tag = " <-- BEST ML" if name == best_name else ""
        acc_s = f"{r['test_accuracy']:.4f}" if r["test_accuracy"] is not None else "  N/A"
        print(f"{name:<22} {acc_s:>9} {r['balanced_accuracy']:>9.4f} {r['macro_f1']:>9.4f}{tag}")

    sign = lambda v: f"+{v:.4f}" if v >= 0 else f"{v:.4f}"
    print(f"\nBest ML model : {best_name}")
    print(f"Rule baseline : TestAcc={rule_acc:.4f}  Macro F1={rule_f1:.4f}")
    print(f"Improvement   : {sign(gain_acc)} accuracy  |  {sign(gain_f1)} macro F1 over rule baseline")

    print(f"\n  CLASSIFICATION REPORT — {best_name} (on test set, Aloe Vera only)")
    print("-" * 54)
    print(results[best_name]["classification_report"])

    print(f"  CONFUSION MATRIX — {best_name}  |  Rows=Actual, Cols=Predicted")
    print(f"  Labels: {LABELS}")
    for row_lbl, cm_row in zip(LABELS, results[best_name]["confusion_matrix"]):
        print(f"  {row_lbl:<12}: {cm_row}")

    print("\n  RULE-BASED BASELINE CLASSIFICATION REPORT")
    print("-" * 54)
    print(rule_report)

    # Feature importance (RF)
    if best_name == "RandomForest":
        imps = best_model.feature_importances_
        print("\n  FEATURE IMPORTANCES (RandomForest, trained on Aloe Vera data)")
        for feat, imp in sorted(zip(FEATURES, imps), key=lambda x: -x[1]):
            bar = "#" * int(imp * 45)
            print(f"  {feat:<22}: {imp:.4f}  {bar}")

    print(f"\n  Saved best model ({best_name}) -> {MODEL_PATH}")

    # ── Save JSON report for /api/model-report endpoint ──────────────────────
    report_data = {
        "best_model":             best_name,
        "dataset": {
            "total_rows":         int(len(df)),
            "aloe_only":          True,
            "excluded_sources":   ["indoor_plant_iot_reference_pool"],
            "class_distribution": df[TARGET].value_counts().to_dict(),
            "train_size":         int(len(train_df)),
            "val_size":           int(len(val_df)),
            "test_size":          int(len(test_df)),
        },
        "improvement_over_rules": {
            "accuracy_gain": gain_acc,
            "macro_f1_gain": gain_f1,
        },
        "results": {
            name: {k: v for k, v in r.items() if k != "classification_report"}
            for name, r in results.items()
        },
    }
    with REPORT_PATH.open("w") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Saved training report -> {REPORT_PATH}")

    return results


if __name__ == "__main__":
    train_and_evaluate()
