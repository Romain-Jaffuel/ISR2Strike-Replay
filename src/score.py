import json
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

def main():
    Path("data").mkdir(parents=True, exist_ok=True)

    with open("data/events_simulated.json", "r", encoding="utf-8") as f:
        events = json.load(f)
    df = pd.DataFrame(events)
    pipe = joblib.load("model/relevance_clf.pkl")

    features = ["cls", "confidence", "priority_hint"]
    df["relevant_prob"] = pipe.predict_proba(df[features])[:, 1]
    df["relevant"] = (df["relevant_prob"] >= 0.5).astype(int)

    kepler_cols = [
        "timestamp", "lat", "lon", "cls",
        "confidence", "priority_hint",
        "relevant_prob", "relevant", "track_id"
    ]
    df[kepler_cols].to_csv("data/kepler_events.csv", index=False)

    # Engagement policy: conservative ROE threshold
    df["engage"] = (
        (df["relevant"] == 1) &
        (df["confidence"] >= 0.75) &
        (df["priority_hint"] <= 2)
    ).astype(int)

    # Aggregates for downstream reporting
    agg = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_events": len(df),
        "detections": int((df["confidence"] >= 0.5).sum()),
        "relevant": int(df["relevant"].sum()),
        "engagement_recommended": int(df["engage"].sum()),
        "positive_ratio": float(df["relevant"].mean()),
        "engage_ratio": float(df["engage"].mean())
    }
    pd.Series(agg).to_json("data/summary.json", indent=2)

    df.to_csv("data/scored_events.csv", index=False)

if __name__ == "__main__":
    main()
