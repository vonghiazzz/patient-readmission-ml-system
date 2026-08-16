# DIAGRAMS & QUICK REFERENCE

## I. EXECUTABLE DIAGRAMS (Copy-paste ready)

### A. FULL SYSTEM ARCHITECTURE

```mermaid
graph TB
    subgraph Input["Input Layer"]
        RAW["data/raw/diabetic_data.csv<br/>(101,766 encounters)"]
        DOC["docs/api/sample_request.json<br/>(40 fields)"]
    end

    subgraph DataOps["Data Processing"]
        ING["src/data/ingestion.py"]
        VALID["src/data/validation.py"]
        SPL["src/data/splitting.py"]
        COHORT["create_huy_cohort()"]
    end

    subgraph Features["Feature Engineering"]
        BUILD["build_huy_features()"]
        FE52["52 model features<br/>7 categorical"]
    end

    subgraph Training["Model Training"]
        SMOTE["SMOTENC<br/>70,543 rows"]
        CB["CatBoost<br/>500 iter, depth 4"]
        THRESH["Threshold optimize<br/>0.8564852"]
    end

    subgraph Artifacts["Frozen Artifacts"]
        PKL["model.pkl"]
        STATE["preprocessing_state.json"]
        MAN["feature_manifest.json"]
        META["metadata.json"]
        REF["reference_predictions.json"]
    end

    subgraph API["API Runtime"]
        MAIN["FastAPI lifespan<br/>load_production_artifacts"]
        ROUTES["GET /health<br/>GET /ready<br/>POST /predict"]
        DEP["ProductionArtifacts<br/>contract validation"]
    end

    subgraph Monitoring["Observability"]
        PROM["Prometheus metrics<br/>HTTP, predictions, ready"]
        ALERT["Alert rules<br/>availability, error rate"]
        GRAF["Grafana dashboards"]
    end

    subgraph Infrastructure["Deployment"]
        DOCKER["Docker container<br/>Python 3.11, non-root"]
        COMPOSE["docker-compose.yml<br/>api + frontend + mlflow"]
    end

    subgraph Frontend["Client"]
        REACT["React app"]
        FORM["Input form (40 fields)"]
        RESULT["Display risk_score"]
    end

    subgraph CI["CI/CD"]
        TEST["pytest (60%+ coverage)"]
        VERIFY["Verify artifacts"]
        BUILD_IMG["Docker build"]
    end

    RAW --> SPL --> COHORT
    COHORT --> SMOTE --> CB --> THRESH

    CB --> PKL
    SPL --> STATE
    FE52 --> MAN
    THRESH --> META
    REF -.-> MAN

    Artifacts --> MAIN
    MAIN --> DEP --> ROUTES

    DOC --> ROUTES
    ROUTES --> PROM --> ALERT
    ALERT --> GRAF

    DEP --> DOCKER --> COMPOSE

    REACT --> FORM --> ROUTES
    ROUTES --> RESULT

    CI -.-> Artifacts

    style PKL fill:#ccffcc
    style ROUTES fill:#ccccff
    style PROM fill:#ffcccc
```

### B. DATA FLOW: 101K → 56K → TRAIN/TEST → SMOTE

```mermaid
graph LR
    A["101,766<br/>encounters"]
    B["Filter:<br/>missing diag<br/>invalid demo<br/>discharge=11"]
    C["First encounter<br/>per patient_nbr"]
    D["Outlier filter<br/>|z| &lt; 3"]
    E["56,653<br/>FINAL"]
    F["Stratified<br/>80/20"]
    G["45,322<br/>TRAIN"]
    H["11,331<br/>TEST"]
    I["SMOTENC<br/>70%"]
    J["70,543<br/>after SMOTE"]

    A -->|~60% remain| B
    B -->|71,518 unique| C
    C -->|remove outliers| D
    D -->|90% pos rate| E
    E -->|stratified| F
    F --> G
    F --> H
    G -->|70% synthetic<br/>29,047 positive| I --> J

    style E fill:#ccffcc
    style J fill:#ffcccc
    style H fill:#ccccff
```

### C. FEATURE ENGINEERING PIPELINE (40 → 52)

```mermaid
graph LR
    A["40 Raw<br/>REQUEST_FEATURES"]

    B1["Gender encode<br/>Female→0, Male→1"]
    B2["Age map<br/>[80-90)→85"]
    B3["Diagnosis group<br/>ICD-9→level-1"]
    B4["Medication encode<br/>No→0, else→1"]

    C1["Service util<br/>log1p transforms"]
    C2["Medication agg<br/>numchange, nummed"]

    D["Interaction<br/>9 features<br/>med×lab, age×diag"]

    E1["Standard scale 1"]
    E2["Min-max scale"]
    E3["Standard scale 2"]
    E4["Age min-max"]

    F["52 Model<br/>MODEL_INPUT_FEATURES"]
    G["CatBoost<br/>predict_proba"]

    A --> B1
    A --> B2
    A --> B3
    A --> B4
    B1 --> C1
    B2 --> C2
    B3 --> C1
    B4 --> C2
    C1 --> D
    C2 --> D
    D --> E1
    E1 --> E2
    E2 --> E3
    E4 -.-> E3
    E3 --> F
    F --> G

    style F fill:#ccffcc
    style G fill:#ffcccc
```

### D. API REQUEST → PREDICTION FLOW

```mermaid
sequenceDiagram
    participant React
    participant API
    participant Pydantic
    participant Features
    participant Model
    participant Response

    React->>API: POST /predict (40 fields JSON)
    API->>Pydantic: Validate schema
    Pydantic->>Pydantic: Check bounds, enums, extra=forbid
    Pydantic->>Features: Pass 40 fields
    Features->>Features: build_huy_features()
    Features->>Features: Encode, group, scale (52 output)
    Features->>Model: 52 features ready
    Model->>Model: predict_proba()
    Model->>Model: risk_score = P(class 1)
    Model->>Model: if risk_score >= 0.8564852<br/>then prediction = 1<br/>else 0
    Model->>Response: Return metrics
    Response->>React: PredictionResponse JSON
    React->>React: Display risk_score + status

    Note over API,Model: Total latency: ~50-100ms
```

### E. MONITORING & ALERT FLOW

```mermaid
graph TB
    API["FastAPI /metrics<br/>endpoint"]

    HTTP["HTTP Metrics"]
    REQ["requests_total"]
    ERR["errors_total"]
    LAT["latency_histogram"]

    PRED["Prediction Metrics"]
    COUNT["predictions_total"]
    RISK["risk_score histogram"]

    READY["Readiness Metrics"]
    MODEL_R["model_ready gauge"]
    INFO["model_info gauge"]

    API --> HTTP
    API --> PRED
    API --> READY

    HTTP --> REQ
    HTTP --> ERR
    HTTP --> LAT

    PRED --> COUNT
    PRED --> RISK

    PROM["Prometheus<br/>scrape:5s"]
    ALERT["Alert Rules<br/>evaluation:5s"]

    REQ --> PROM
    ERR --> PROM
    LAT --> PROM
    COUNT --> PROM
    RISK --> PROM
    MODEL_R --> PROM
    INFO --> PROM

    PROM --> ALERT

    A1["API Unavailable<br/>(up == 0, 1m)"]
    A2["Model Not Ready<br/>(model_ready == 0, 1m)"]
    A3["Error Rate High<br/>(> 5%, 5m)"]
    A4["Latency High<br/>(P95 > 1s, 5m)"]

    ALERT --> A1
    ALERT --> A2
    ALERT --> A3
    ALERT --> A4

    GRAF["Grafana Dashboard"]

    PROM --> GRAF

    style PROM fill:#ccffcc
    style ALERT fill:#ffcccc
    style GRAF fill:#ccccff
```

### F. CI/CD PIPELINE

```mermaid
graph LR
    PR["PR created"]
    GH["GitHub Actions<br/>trigger"]

    V1["Verify artifacts<br/>model.pkl exists?<br/>preprocessing_state.json?"]
    V2["Install deps<br/>pip check"]
    V3["Lint & format<br/>ruff check<br/>ruff format --check"]
    V4["Tests<br/>pytest coverage ≥ 60%"]
    V5["Docker build<br/>docker build"]

    PASS["✓ All pass"]
    MERGE["✓ Merge allowed"]

    FAIL["✗ Fail"]
    BLOCK["✗ Merge blocked"]

    PR --> GH
    GH --> V1
    V1 -->|✓| V2
    V1 -->|✗| FAIL
    V2 -->|✓| V3
    V2 -->|✗| FAIL
    V3 -->|✓| V4
    V3 -->|✗| FAIL
    V4 -->|✓| V5
    V4 -->|✗| FAIL
    V5 -->|✓| PASS
    V5 -->|✗| FAIL
    PASS --> MERGE
    FAIL --> BLOCK

    style MERGE fill:#ccffcc
    style BLOCK fill:#ffcccc
```

### G. MODEL LIFECYCLE

```mermaid
graph TB
    NB["Huy's notebook<br/>huy-catboost-1.0.0"]
    PKL["model.pkl<br/>SHA256: a0d11b7e..."]
    FREEZE["Frozen artifacts<br/>preprocessing_state.json<br/>feature_manifest.json<br/>metadata.json"]

    TEST["Regression test<br/>0.6643 → 0<br/>0.9379 → 1"]

    CI["CI verifies"]
    DOCKER["Docker build<br/>COPY model.pkl"]
    COMPOSE["docker-compose up<br/>api + frontend"]

    START["Container startup<br/>lifespan()"]
    LOAD["load_production_artifacts()"]
    VALIDATE["Validate contract"]

    READY["model_ready = 1<br/>/ready → 200"]
    NOTREADY["model_ready = 0<br/>/ready → 503"]

    PREDICT["POST /predict<br/>40 fields"]
    SERVE["predict_proba()"]
    RESPONSE["risk_score + prediction"]

    MON["Prometheus collect"]
    ALERT["Alert if issues"]

    NB --> PKL
    PKL --> FREEZE
    FREEZE --> TEST
    TEST --> CI
    CI --> DOCKER
    DOCKER --> COMPOSE
    COMPOSE --> START
    START --> LOAD
    LOAD --> VALIDATE
    VALIDATE -->|✓| READY
    VALIDATE -->|✗| NOTREADY
    READY --> PREDICT
    PREDICT --> SERVE
    SERVE --> RESPONSE
    RESPONSE --> MON
    MON --> ALERT

    style READY fill:#ccffcc
    style NOTREADY fill:#ffcccc
    style RESPONSE fill:#ccccff
```

---

## II. COMPARISON TABLE: TRAINING vs EVALUATION vs SERVING

| Phase | Data | Process | Artifacts | Goal |
| --- | --- | --- | --- | --- |
| **Training** | 70,543 (after SMOTE) | CatBoost fit 500 trees, depth 4, class_weights | model.pkl, preprocessing_state.json, metadata.json | Learn pattern from imbalanced data |
| **Evaluation** | 11,331 (test holdout) | predict_proba() + threshold 0.8564852, confusion matrix | evaluation_metrics.json, calibration_curve.csv | Measure performance (recall 5%, ROC 0.567) |
| **Serving** | 1 sample at a time (inference request) | build_huy_features() 40→52, predict_proba(), threshold | model.pkl (loaded), preprocessing_state.json (frozen) | Return risk_score to client |

---

## III. FEATURE CONTRACT MAPPING

**Request → Model transformation:**

| Index | Request Feature | Type | Transform | Model Feature |
| --- | --- | --- | --- | --- |
| 0 | race | categorical | Str pass-through | race |
| 1 | gender | categorical | {Female→0, Male→1} | gender |
| 2 | age | categorical | {[80-90)→85, ...} | age |
| ... | ... | ... | ... | ... |
| 14 | max_glu_serum | categorical | Abnormal binary | max_glu_serum |
| 15 | A1Cresult | categorical | Abnormal binary | A1Cresult |
| 16-35 | medications (20) | categorical | {No→0, Up→1, Down→1, Steady→1} | medications |
| 36 | change | categorical | {No→0, Ch→1} | change |
| 37 | diabetesMed | categorical | {No→0, Yes→1} | diabetesMed |
| 38 | diag_1 | string | ICD-9 group → [0,8] | level1_diag1 |
| 39 | (extra fields removed) | | | |
| **NEW** | numchange | derived | Count of Up/Down | numchange |
| **NEW** | nummed | derived | Count != No | nummed |
| **NEW** | service_utilization_log1p | derived | log1p(out+emerg+inp) | service_utilization_log1p |
| ... | ... | ... | ... | ... |
| **52** | interactions (9) | derived | med×lab, age×diag, ... | Various |

**Categorical indices [0, 3, 4, 5, 11, 12, 37]:**
- 0: race
- 3: admission_type_id
- 4: discharge_disposition_id
- 5: admission_source_id
- 11: max_glu_serum
- 12: A1Cresult
- 37: level1_diag1

---

## IV. THRESHOLD VISUALIZATION

```
Risk Score Distribution (test set)
│
│  ▁▂▃▄▅▆▇█ (histograms represent continuous distribution)
│  ▃▆████▅▂
│  ▅██████▃
│  ▂█████▅▂▁
│  ▁████▂▁
└──────────────────────────────────→ Risk Score (0 to 1)
   0   0.25  0.5   0.75  0.8564852  1.0

   Threshold = 0.8564852

   Before: predict 0 (not high risk)
   ↓ threshold
   ↓
   After: predict 1 (high risk)
```

**Interpretation:**
- Threshold 0.5: Default (50% confidence needed)
- Threshold 0.8564852: High confidence (85.6% needed)
  - Fewer false positives (better precision)
  - More false negatives (worse recall) ← recall 5.13%

---

## V. ERROR MATRIX INTERPRETATION

```
Test set (11,331 total)

         Predicted 0      Predicted 1
         (no readmit)     (readmit)

Actual 0
(no):    10,115 TN        260 FP        = 10,375 actual negative
         (correct)        (false alarm)

Actual 1
(<30):   907 FN           49 TP         = 956 actual positive
         (missed)         (caught)

         11,022           309
         predicted neg    predicted pos

KEY METRICS:
─────────────────────────────────────
Sensitivity (Recall) = TP/(TP+FN) = 49/956 = 5.1% ← LOW
Specificity = TN/(TN+FP) = 10115/10375 = 97.5% ← HIGH

Precision = TP/(TP+FP) = 49/309 = 15.9% ← LOW
False Positive Rate = FP/(FP+TN) = 260/10375 = 2.5% ← LOW

F1 Score = 2 × (Prec × Rec) / (Prec + Rec) = 7.7% ← VERY LOW

Interpretation:
─────────────────────────────────────
✗ Model very BAD at finding readmissions (5% recall)
✓ Model GOOD at ruling out non-readmission (97% specificity)
✗ When model predicts readmission, usually wrong (16% precision)
→ NOT suitable for clinical screening
```

---

## VI. DATA SANITY CHECKS (Quick validation)

**Pass these checks before presentation:**

```
✓ Model file exists
  ls -la models/production_huy/model.pkl

✓ SHA256 matches
  sha256sum models/production_huy/model.pkl
  → a0d11b7ed0c1956d10afbfda360ec24ae2c55f6d6d50d32ed50780a81160331b

✓ Metadata valid
  cat models/production_huy/metadata.json | grep decision_threshold
  → 0.8564852152742759

✓ Reference predictions match
  python -c "
  from src.api.dependencies import load_production_artifacts
  artifacts = load_production_artifacts(Path('models/production_huy'))
  print(artifacts.model_version)
  print(artifacts.decision_threshold)
  print(artifacts.model_sha256)
  "

✓ Tests pass
  pytest tests/production/ -v

✓ API starts
  python -m uvicorn src.api.main:app --port 8000
  curl http://127.0.0.1:8000/ready
  → {"status": "ready", ...}

✓ Prediction works
  curl -X POST http://127.0.0.1:8000/predict \
    -H 'Content-Type: application/json' \
    --data @docs/api/sample_request.json
  → {"model_version": "huy-catboost-1.0.0", "risk_score": 0.6643..., ...}
```

---

## VII. ONE-PAGE EXECUTIVE SUMMARY

### Patient Readmission ML System

**What:** Reproduce Huy's CatBoost model predicting 30-day hospital readmission risk. Serve as production system with monitoring & tests.

**Data:**
- Raw: 101,766 encounters → Final: 56,653 (44% filtered)
- Train: 45,322 (91.6% negative) → SMOTE → 70,543 (59.3% negative, 40.7% synthetic positive)
- Test: 11,331 (91.5% negative) — NO SMOTE

**Model:**
- Algorithm: CatBoost Classifier
- Features: 40 raw → 52 engineered (encoding, diagnosis grouping, interactions, log transforms, scaling)
- Hyperparameters: 500 trees, depth 4, learning_rate 0.0185, class_weights {0:1, 1:10}
- Threshold: 0.8564852 (optimized via 6-fold CV, not default 0.5)

**Performance (Test):**
| Metric | Value | Verdict |
| --- | --- | --- |
| Recall | 5.1% | WEAK |
| Precision | 15.9% | WEAK |
| ROC-AUC | 0.567 | WEAK |
| PR-AUC | 0.108 | WEAK |
| Brier | 0.386 | NOT calibrated |

**Infrastructure:**
- API: FastAPI, Pydantic validation (40-field schema, extra=forbid)
- Docker: Non-root user, healthcheck, frozen model (no retrain)
- Monitoring: Prometheus + Grafana, 4 alert rules (availability, errors, latency)
- CI/CD: GitHub Actions, artifact verification, pytest (60%+ coverage), Docker build
- Tests: Production contract tests, reference predictions exact to ±1e-12

**Limitations (Documented):**
1. Recall 5.13% → misses 90% of readmissions → NOT clinical production
2. Data leakage → preprocessing fit before split
3. SMOTENC before CV → synthetic samples may leak
4. Probability uncalibrated → Brier 0.386
5. Threshold optimized on training folds (potential overfitting)

**Verdict:**
- **MLOps engineering:** Excellent (reproducible, contract-driven, comprehensive testing, transparent)
- **Model performance:** Weak (recall 5%, ROC 0.567)
- **Suitable for:** Course demonstration, learning MLOps patterns
- **NOT suitable for:** Clinical production without major retraining

---

## VIII. "IF I FORGET" RECOVERY CHECKLIST

During presentation, if you forget something:

**Numbers:**
- [ ] Have cheat sheet with 101k/56k/45k/70k/11k
- [ ] Have metadata.json open for exact threshold
- [ ] Have MODEL_CARD.md for exact metrics

**Diagrams:**
- [ ] Have data flow diagram (101k → 56k → split → SMOTE)
- [ ] Have feature engineering steps (40 → 52)
- [ ] Have API flow (40 fields → 52 features → predict → response)

**Answers:**
- [ ] "101k → 56k" = filters + first encounter
- [ ] "70k" = SMOTENC: 41,496 + 29,047 synthetic
- [ ] "Recall 5%" = misses 90% readmissions
- [ ] "Threshold 0.8564852" = optimized CV, not 0.5
- [ ] "Data leakage" = preprocessing fit before split (documented)

**If examiner asks something you don't know:**
- Say: "That's a great question. Let me check the code/docs to be precise."
- (Reference AUDIT_COMPLETE.md, MODEL_CARD.md, metadata.json, code)
- Never guess; accuracy > speed

---

END OF DIAGRAMS & REFERENCE
