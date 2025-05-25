 # FraudScope360

**End-to-End Fraud Detection Pipeline**

## 🚀 Overview

FraudScope360 is a modular, end-to-end fraud detection system that:

1. Ingests transactions (live from Kafka or batch CSV).
2. Extracts behavioral, network, and text features.
3. Runs five independent detection modules:

   * **Anomaly Detection** (Isolation Forest)
   * **Change-Point Detection** (PELT)
   * **Network Analysis** (Node2Vec)
   * **Identity Clustering** (DBSCAN + Fuzzy Matching)
   * **NLP** (TF-IDF + Logistic Regression)
4. Fuses module outputs into a single `risk_score`.
5. Exposes a **FastAPI** backend (`POST /score`) and a **Streamlit** dashboard.

> **Streamlit UI:** [http://localhost:8501/](http://localhost:8501/)
> **Swagger UI:** [http://127.0.0.1:8000/docs#/default/health\_check\_health\_get](http://127.0.0.1:8000/docs#/default/health_check_health_get)

---

## 📦 Repository Structure

```text
fraudscope360/
├── api/                   # FastAPI service
│   └── main.py            # API endpoints
├── configs/               # YAML configuration files
│   └── config.yaml        # Module parameters & weights
├── data_ingest/           # Simulation & ingestion code
│   └── simulate.py        # simulate(), simulate_tickets(), simulate_edges()
├── dashboard/             # Streamlit app
│   └── streamlit_app.py   # Dashboard frontend
├── features/              # Feature engineering
│   └── feature_engineering.py
├── modules/               # Detection modules
│   ├── anomaly/           # IsolationForest
│   ├── changepoint/       # PELT
│   ├── graph/             # Node2Vec
│   ├── identity/          # DBSCAN
│   └── nlp/               # TF-IDF + Logistic
├── tests/                 # pytest unit & smoke tests
├── docker-compose.yml     # Full stack orchestration
├── Dockerfile (root)      # (optional) combined build
├── fraud_detection_app.py # Standalone demo script
└── README.md              # This documentation
```

---

## 🏁 Quickstart (Docker Compose)

Ensure you have **Docker** and **Docker Compose** installed.

```bash
# 1. Clone the repo
git clone git@github.com:Adya6714/FraudScope360.git
cd FraudScope360

# 2. Start all services (Zookeeper, Kafka, API, Dashboard)
docker-compose up --build -d

# 3. Visit the UIs:
#    - Streamlit Dashboard: http://localhost:8501/
#    - FastAPI Swagger:    http://127.0.0.1:8000/docs

# 4. (Optional) Tail logs
docker-compose logs -f api
docker-compose logs -f dashboard
```

---

## 🖥️ Local Python Setup

Alternatively, run locally without Docker:

```bash
# 1. Create & activate virtualenv
python3 -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start FastAPI
uvicorn api.main:app --reload

# 4. Start Streamlit
tcd dashboard && streamlit run streamlit_app.py
```

---

## 🔧 Configuration

All module parameters and fusion weights live in `configs/config.yaml`. Edit thresholds, penalties, and weights without changing code.

```yaml
anomaly:
  contamination: 0.02
  random_state: 42
changepoint:
  model: "rbf"
  pen: 10
graph:
  dimensions: 32
  walk_length: 10
  num_walks: 50
  window: 5
identity:
  eps: 1
  min_samples: 1
nlp:
  ngram_range: [1, 2]
  C: 1.0
fusion_weights:
  anomaly: 0.3
  change_point: 0.2
  network: 0.2
  id_cluster: 0.15
  nlp: 0.15
```

---

## 📝 Example Transaction Walkthrough

Below is an example transaction we’ll score via both UIs.

```jsonc
{
  "user": "user_101",
  "amount": 9523.47,
  "timestamp": "2025-06-01T02:17:45",
  "device": "device_23",
  "ip": "203.0.113.54"
}
```

### A) Using Swagger UI (Backend)

1. Open the Swagger page:
   `http://127.0.0.1:8000/docs#/default/score_score_post`
2. Click **Try it out** on `POST /score`.
3. Paste the JSON above into the request body.
4. Click **Execute**.
5. Inspect the JSON response:

```json
{
  "risk_score": 0.82,
  "breakdown": {
    "anomaly": 0.95,
    "change_point": 0,
    "network": 0.65,
    "id_cluster": 0,
    "nlp": 0.27
  }
}
```

### B) Using Streamlit Dashboard (Frontend)

1. Open: [http://localhost:8501/](http://localhost:8501/)
2. In the **New Transaction** sidebar, fill in:

   * **User ID**: `user_101`
   * **Amount**: `9523.47`
   * **Timestamp**: `2025-06-01T02:17:45`
   * **Device ID**: `device_23`
   * **IP Address**: `203.0.113.54`
3. Click **Score Transaction**.
4. View the rendered **Risk Score** and **Breakdown** below.

> **Streamlit UI** is for interactive, human-friendly analytics, while **Swagger UI** is for API testing and integration. Both drive the same backend logic.

---

✅ Testing

Run unit and smoke tests via pytest:

pytest -q

📂 Packaging & Deployment

Docker Compose orchestrates Zookeeper, Kafka, API, and Dashboard.

SSH → GitHub for code: git@github.com:Adya6714/FraudScope360.git


