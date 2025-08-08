import json, random
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, average_precision_score, roc_auc_score,
    precision_recall_curve, f1_score
)
import joblib

random.seed(0); np.random.seed(0)

def heuristic_label(row):
    if row["cls"] in ["uav", "usv", "small_boat"] and row["confidence"] >= 0.6 and row["priority_hint"] <= 3:
        return 1
    if row["cls"] in ["unknown", "merchant_vessel"] and row["confidence"] >= 0.65 and row["priority_hint"] <= 2:
        return 1
    return 0

def pick_threshold(y_true, y_prob, min_precision=0.8):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    cand = [t for pp, rr, t in zip(p[:-1], r[:-1], thr) if pp >= min_precision]
    if cand:
        return float(max(cand, key=lambda t: r[list(thr).index(t)]))
    f1_best, thr_best = -1, 0.5
    for pp, rr, t in zip(p[:-1], r[:-1], thr):
        f1 = 0.0 if (pp + rr) == 0 else 2 * pp * rr / (pp + rr)
        if f1 > f1_best:
            f1_best, thr_best = f1, float(t)
    return thr_best

def main():
    Path("model").mkdir(parents=True, exist_ok=True)

    with open("data/events_simulated.json", "r") as f:
        events = json.load(f)
    df = pd.DataFrame(events)
    df["label"] = df.apply(heuristic_label, axis=1).astype(int)

    features_cat = ["cls"]
    features_num = ["confidence", "priority_hint"]
    groups = df["track_id"]

    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features_cat),
        ("num", StandardScaler(), features_num),
    ])

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )

    pipe = Pipeline([
        ("prep", preproc),
        ("clf", clf)
    ])

    param_grid = {
        "clf__C": [0.25, 0.5, 1.0, 2.0],
        "clf__penalty": ["l1", "l2"]
    }

    # Group-aware split to prevent leakage across trajectories
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(df, df["label"], groups))
    dtr, dte = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()

    Xtr, ytr = dtr[features_cat + features_num], dtr["label"].values
    Xte, yte = dte[features_cat + features_num], dte["label"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    grid.fit(Xtr, ytr)

    best_model = grid.best_estimator_
    y_prob = best_model.predict_proba(Xte)[:, 1]
    threshold = pick_threshold(yte, y_prob, min_precision=0.8)
    y_pred = (y_prob >= threshold).astype(int)

    ap = average_precision_score(yte, y_prob)
    auc = roc_auc_score(yte, y_prob)
    f1 = f1_score(yte, y_pred)

    print(f"Best params: {grid.best_params_}")
    print(f"Val AP: {ap:.3f} | ROC-AUC: {auc:.3f} | F1@thr={threshold:.3f}: {f1:.3f}")
    print(classification_report(yte, y_pred, digits=3))

    joblib.dump(best_model, "model/relevance_clf.pkl")
    meta = {
        "version": 1,
        "features": features_cat + features_num,
        "threshold": threshold,
        "metrics": {"AP": float(ap), "ROC_AUC": float(auc), "F1_at_thr": float(f1)},
        "best_params": grid.best_params_,
        "train_size": int(len(dtr)),
        "val_size": int(len(dte)),
        "positive_ratio_train": float(dtr["label"].mean()),
        "positive_ratio_val": float(dte["label"].mean()),
        "notes": "Group split by track_id; balanced class weights."
    }
    with open("model/metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
