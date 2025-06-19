# SustainaML AutoML

A lightweight, **energy‑aware AutoML** toolkit.  A Flask back‑end trains multiple ML frameworks while tracking CO₂/energy with , and a Streamlit front‑end lets you explore results, feature importance, and hyper‑parameter impact — all in your browser.

# Table of Contents

1. Features

2. Project Structure

3. Quick Start

4. API Reference

5. Package Versions

6. Troubleshooting

7. Acknowledgement



## 1. Features

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

## 3. Quick Start

**1 · Clone & create env**

git clone <your‑repo‑url> sustainaml-automl

cd sustainaml-automl

python -m venv venv

source venv/bin/activate   # Windows: venv\Scripts\activate_
**2 · Install requirements**

pip install -r requirements.txt

#OR – for the exact versions we tested

#pip install -r requirements_locked.txt

**3 · Run the stack**

In two terminals or two split panes 

#Terminal 1 – REST back‑end

python backend.py

#Terminal 2 – UI front‑end

streamlit run frontend.py   # or streamlit run app.py

Visit the URL Streamlit prints (default http://localhost:8501).  Upload a CSV where **the last column is the target label.**

## 4. API Reference

POST /run_automl

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
  
}

Returns per‑algorithm metrics, carbon/energy figures, hyper‑params & feature importances.

## 5. Package Versions


![image](https://github.com/user-attachments/assets/fceed1dc-ef9a-467f-b0f2-465051cbd072)


## 6.  Troubleshooting

**Symptom**                                                                 **Fix**      


-**ModuleNotFoundError:**                   Re‑run pip install -r requirements.txt inside the activated venv


- **Tracker raises No GPU found:**          Ignore – CodeCarbon falls back to CPU measurement
  

- **Streamlit shows blank page:**              Refresh browser; check that backend.py is still running
  
## 7. Acknowledgements

CodeCarbon Library for carbon estimation  & the open‑source community • 🌱
