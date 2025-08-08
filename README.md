# ISR to Strike Replay

## Project Description
This project simulates, scores, and visualizes ISR (Intelligence, Surveillance, Reconnaissance) events to assess engagement recommendations ("strike recommendations").  
It generates synthetic moving objects (ships, UAVs, USVs, unknown contacts) within a defined geographic area, assigns them kinematic and environmental attributes, and applies a scoring model to determine which events are considered "relevant" for action.

The goal is twofold:
1. **Generate realistic synthetic ISR-like trajectories and detections.**
2. **Produce analytical and visual outputs** with interactive maps (Kepler.gl) and statistical plots (Matplotlib).

---

## Features
- Route simulation for maritime and aerial targets with kinematic noise.
- Generation of spurious detections to test filtering robustness.
- Class, speed, priority, and confidence assignment using probabilistic rules.
- Calculation of key metrics: detections, relevance estimation, engagement recommendations.
- Export:
  - CSV and JSON event datasets.
  - Interactive HTML map (Kepler.gl).
  - Markdown report with embedded figures.

---

## Installation
1. Clone the repository:
    git clone <https://github.com/Romain-Jaffuel/ISR2Strike-Replay.git>
    cd <PROJECT_FOLDER>

2. Create a Python 3.12 virtual environment:
    python -m venv venv312
    venv312\Scripts\activate

3. Install dependencies:
    pip install -r requirements.txt

## Usage
1. Generate simulated events:
    python src/simulate_events.py

2. Train Relevance:
    python src/train_relevance.py

3. Score and filter events:
    python src/score_events.py

4. Build the report and interactive map:
    python src/build_report.py

## Outputs
- Markdown report: out/report.md
- Figures:
    - out/figs/detections_by_class.png
    - out/figs/confidence_hist.png
- Interactive map: out/kepler_map.html

## Main Dependencies
- Python 3.12
- NumPy
- Pandas
- Matplotlib
- Kepler.gl

## Notes
- All data is synthetic and does not represent any real-world scenario.
- The current scoring model is heuristic-based and should be replaced with a domain-specific or trained model for operational use.