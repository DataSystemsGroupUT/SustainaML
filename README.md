### SustainaML AutoML

A lightweight, **energy‑aware AutoML** toolkit.  A Flask back‑end trains multiple ML frameworks while tracking CO₂/energy with , and a Streamlit front‑end lets you explore results, feature importance, and hyper‑parameter impact — all in your browser.

#Table of Contents

1. Features

2. Project Structure

3. Quick Start

4. API Reference

5. Package Versions

6. Troubleshooting

7. Acknowledgement



# 1. Features

- **AutoML search across frameworks** — FLAML, “H2O”, MLJAR

- **Time‑budgeted training** with incremental / warm‑start loops

- **Energy & CO₂ tracking** via CodeCarbon (offline mode)

- **Interactive Streamlit UI**
   ▸ upload CSV  ▸ pick frameworks & algorithms  ▸ edit hyper‑parameters  ▸ visualise leaderboards, scatter plots, feature importance, pipeline diagrams, time‑budget trade‑offs

- **REST API** (/run_automl) for programmatic access

# 2. Project Structure
├── backend.py      # Flask API – training & carbon tracking
├── frontend.py     # Main Streamlit interface (rich UI)
├── app.py          # Minimal Streamlit demo (optional)
└── README.md       # You are here
**Tip :** frontend.py and app.py do not share state; keep one running at a time.
# 3. Quick Start
**1 · Clone & create env**
_git clone <your‑repo‑url> sustainaml-automl
cd sustainaml-automl
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate_
**2 · Install requirements**
_pip install -r requirements.txt
#OR – for the exact versions we tested
#pip install -r requirements_locked.txt_
**3 · Run the stack**
In two terminals or two split panes 
_#Terminal ① – REST back‑end
python backend.py

#Terminal ② – UI front‑end
streamlit run frontend.py   # or streamlit run app.py_
Visit the URL Streamlit prints (default http://localhost:8501).  Upload a CSV where **the last column is the target label.**
# 4. API Reference
_POST /run_automl
{
  "frameworks": ["FLAML", "H2O", "MLJAR"],
  "algorithms": {
      "FLAML": {"RF": true, "XGBoost": true},
      "H2O":   {"GLM": true}
      ..............................
  },
  "hyperparams": {
      "FLAML": {"RF": {"n_estimators": 150}}
      .................................
  },
  "time_budget":10, 30, 60, 120
  "data": "<pandas.DataFrame>.to_json()"   // last col = y
}_
Returns per‑algorithm metrics, carbon/energy figures, hyper‑params & feature importances.
# 5. Package Versions
Below are the tested versions.  Newer releases usually work, but lock these for reproducibility.

_Python              3.10+

# Core
flask               3.0.2
streamlit           1.34.0
requests            2.32.2

# Data / ML
pandas              2.2.2
numpy               1.26.4
scikit‑learn        1.5.0
flaml               2.1.5
xgboost             2.0.5
lightgbm            4.3.0
catboost            1.2.3
codecarbon          2.3.0

# Visualisation
plotly              5.21.0
matplotlib          3.9.0
seaborn             0.13.2_
# 6.  Troubleshooting
**Symptom**                                                                 **Fix  **      

- ModuleNotFoundError                     Re‑run pip install -r requirements.txt inside the activated venv

- Tracker raises No GPU found             Ignore – CodeCarbon falls back to CPU measurement

- Streamlit shows blank page             Refresh browser; check that backend.py is still running
- 
# 7. Acknowledgements
CodeCarbon Library for carbon estimation  & the open‑source community • 🌱
