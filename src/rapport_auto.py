import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import webbrowser
from keplergl import KeplerGl

def main():
    Path("out/figs").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/scored_events.csv", parse_dates=["timestamp"])
    with open("data/summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    by_class = df.groupby("cls").size().sort_values(ascending=False)
    by_class.plot(kind="bar", title="Detections by class")
    plt.tight_layout()
    plt.savefig("out/figs/detections_by_class.png")
    plt.close()

    df["confidence"].plot(kind="hist", bins=20, title="Confidence distribution")
    plt.tight_layout()
    plt.savefig("out/figs/confidence_hist.png")
    plt.close()

    rec = df[df["engage"] == 1].sort_values("timestamp")
    top_lines = min(10, len(rec))
    bullet = "\n".join([
        f"- {r.timestamp} | {r.cls} | conf={r.confidence:.2f} | prio={r.priority_hint} | prob={r.relevant_prob:.2f}"
        for _, r in rec.head(top_lines).iterrows()
    ])

    md = f"""# ISR to Strike Replay - Summary Brief

**Time window:** {df['timestamp'].min()} to {df['timestamp'].max()}

## Key numbers
- Total events: {summary['total_events']}
- Detections (conf>=0.5): {summary['detections']}
- Relevant (model): {summary['relevant']}
- Engagement recommended: {summary['engagement_recommended']}

## Rationale
Engagement is recommended when an event is predicted relevant (prob>=0.5), confidence>=0.75, and priority<=2.

## Quick view of first recommended engagements
{bullet if bullet else "- None"}

## Figures
![Detections by class](figs/detections_by_class.png)

![Confidence distribution](figs/confidence_hist.png)

## Notes
- Relevance is learned from a synthetic heuristic; replace with domain rules or labeled data.
- CSV `data/kepler_events.csv` is ready for Kepler.gl (lat, lon, timestamp).
"""
    Path("out").mkdir(exist_ok=True, parents=True)
    with open("out/report.md", "w", encoding="utf-8") as f:
        f.write(md)

    kepler_df = pd.read_csv("data/kepler_events.csv")
    map_ = KeplerGl(height=600, data={"Events": kepler_df})

    html_path = Path("out/kepler_map.html")
    map_.save_to_html(file_name=str(html_path), read_only=True)

if __name__ == "__main__":
    main()
