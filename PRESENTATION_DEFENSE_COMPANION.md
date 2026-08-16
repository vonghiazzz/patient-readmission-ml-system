# PRESENTATION & DEFENSE COMPANION

## PHẦN A: PRESENTATION SCRIPT FULL VERSION (Chi tiết từng phần)

### PRE-PRESENTATION (Chuẩn bị trước 5 phút)

**Mental checklist:**
- [ ] Biết số 101,766 / 56,653 / 45,322 / 70,543 / 11,331
- [ ] Nhớ recall 5.13%, ROC-AUC 0.567
- [ ] Nhớ threshold 0.8564852 (không phải 0.5)
- [ ] Hiểu flow: data → cohort → split → SMOTE → CatBoost → threshold → test metrics
- [ ] Hiểu 40→52 feature engineering
- [ ] Nhớ API /health vs /ready khác nhau
- [ ] Có thể vẽ từng flow nếu cần

### PHẦN 1: OPENING (0:00–0:45)

**Kịch bản:**

> "Good morning. I will present our Patient Readmission ML System—a course demonstration of MLOps best practices.
>
> This project achieves two things:
> 1. **Reproduce exactly Huy's final CatBoost model** from his notebook using rigorous data engineering
> 2. **Serve it as a production system** with Docker, monitoring, and automated tests
>
> The goal is NOT to build the best model—it's to demonstrate how to build, test, serve, and monitor ML systems correctly."

**Sẵn sàng cho câu hỏi:**
- "What's the business use case?" → Course demo, not actual clinical
- "Why not train a better model?" → Frozen reproduction first, improvements later
- "Is this for hospital deployment?" → No, learning demonstration

### PHẦN 2: PROBLEM STATEMENT (0:45–1:30)

**Kịch bản:**

> "We're predicting 30-day hospital readmission risk for diabetic patients.
>
> **Target definition:**
> - Readmission < 30 days → class 1 (positive, high risk)
> - Readmission > 30 days or no readmission → class 0 (negative, low risk)
>
> **Success criteria:**
> - Reproducibility: predict exact same scores across reloads
> - Production readiness: serve frozen model reliably
> - Monitoring: track performance, latency, errors
> - Transparency: document limitations, not hide them
>
> This is NOT about clinical validation—it's about engineering rigor."

**Sẵn sàng cho câu hỏi:**
- "Why 30 days specifically?" → Arbitrary threshold for demo
- "Is model used clinically?" → No, course demo only
- "Any ethical concerns?" → Yes, documented—model not suitable for autonomous decisions

### PHẦN 3: DATA & COHORT (1:30–2:45)

**Kịch bản:**

> "Our raw dataset has **101,766 hospital encounters** from the UCI diabetes dataset.
>
> But not all encounters are usable. Huy applied strict cohort filters:
>
> **Step 1: Eliminate missing or invalid data**
> - Diagnosis codes must be present: diag_1 ≠ '?', diag_2 ≠ '?', diag_3 ≠ '?'
> - Race must be valid: race ≠ '?'
> - Gender must be valid: gender ≠ 'Unknown/Invalid'
> - Discharge disposition: discharge_disposition_id ≠ 11
>
> Result: ~63,000 encounters survive
>
> **Step 2: One encounter per patient**
> - Policy: Keep FIRST encounter per patient_nbr
> - Why? Avoid information leakage—don't let model see patient's history repeated
>
> Result: ~71,518 → 56,653 unique patients
>
> **Step 3: Remove statistical outliers**
> - Filter: |z-score| ≤ 3 on scaled features
> - Remove extreme cases that might be data errors
>
> **Final cohort: 56,653 encounters**
>
> **Distribution:**
> - Negative (no readmission or >30 days): 52,827
> - Positive (readmission <30 days): 3,826
> - Imbalance: 92.5% negative, 7.5% positive"

**数据降低的原因细分:**
- Filter diag/race/gender/disposition: -38,000
- First encounter policy: some patients repeat, so 71k unique
- Outliers: -200
- **Total drop: 101,766 → 56,653 (44% filtered)**

**Sẵn sàng:**
- "Why so much filtering?" → Data quality, leakage prevention
- "Is first encounter policy correct?" → Code enforces drop_duplicates keep="first"; chronological order not verified
- "How many patients were excluded?" → ~15k patient encounters dropped

### PHẦN 4: TRAIN-TEST SPLIT (2:45–3:30)

**Kịch bản:**

> "We split cohort **stratified 80-20** with random_state=42.
>
> **Before SMOTENC:**
> - Training: 45,322 rows
>   - Negative: 41,496 (91.6%)
>   - Positive: 3,826 (8.4%)
>
> - Test: 11,331 rows
>   - Negative: 10,375 (91.5%)
>   - Positive: 956 (8.5%)
>
> **Stratified split ensures:**
> - Positive rate identical in train and test
> - No patient overlap (already guaranteed by first-encounter policy)
>
> **Important:** Test set NOT resampled (only training gets SMOTENC)."

**Sẵn sàng:**
- "Why stratified?" → Ensure positive rate same across folds
- "Why random_state=42?" → Reproducibility
- "What if random_state different?" → Different split, different model

### PHẦN 5: FEATURE ENGINEERING (3:30–4:30)

**Kịch bản:**

> "API receives **40 raw encounter fields**. We transform to **52 model-ready features**.
>
> **Transformation pipeline (all deterministic, frozen):**
>
> **1. Categorical encoding:**
> - gender: {Female → 0, Male → 1}
> - age: '[80-90)' → 85.0 (midpoint of bracket)
> - change: {No → 0, Ch → 1} (medication changed?)
> - diabetesMed: {No → 0, Yes → 1}
>
> **2. Diagnosis grouping:**
> - diag_1 (ICD-9 code) → 9 level-1 groups
> - Examples: V/E codes → 0, Cardiac (390-459) → 1, Respiratory (460-519) → 2
> - level1_diag1 feature (categorical)
>
> **3. Medication aggregate:**
> - 20 medication features, each 'Up', 'Down', 'Steady', 'No'
> - numchange: count of Up/Down changes
> - nummed: count of medications used (≠ No)
> - Each medication: encoded 0/1 (No → 0, else → 1)
>
> **4. Service utilization (log transforms):**
> - number_emergency_log1p = log1p(number_emergency)
> - number_outpatient_log1p = log1p(number_outpatient)
> - service_utilization_log1p = log1p(outpatient + emergency + inpatient)
> - number_inpatient_log1p = log1p(number_inpatient)
>
> **5. Interaction features (9 total):**
> - time_in_hospital × num_lab_procedures
> - num_medications × num_lab_procedures
> - num_medications × number_diagnoses
> - age × number_diagnoses
> - change × num_medications
> - number_diagnoses × time_in_hospital
> - num_medications × log(time_in_hospital)
> - num_medications × log1p(num_procedures)
> - num_medications × log1p(numchange)
>
> **6. Scaling (frozen from notebook):**
> - Standard scaling (z-norm) on features 1 and 2
> - Min-max scaling (0-1) on interactions
> - Standard scaling again on log features
> - Min-max scaling on age
>
> **Result: 52 model features ready for CatBoost**"

**Sẵn sàng:**
- "Why so many transformations?" → Huy's notebook design (reproducibility goal)
- "Are scaling parameters fitted?" → No, frozen from training set
- "Can we skip interactions?" → Not if we're reproducing Huy (model expects 52, not 43)
- "What if we change feature order?" → CatBoost fails—feature_names_ validation

### PHẦN 6: HANDLING IMBALANCE (4:30–5:15)

**Kịch bản:**

> "Training set is highly imbalanced: 91.6% negative, 8.4% positive.
>
> We use **SMOTENC** to balance:
>
> **SMOTE (Synthetic Minority Over-sampling) parameters:**
> - sampling_strategy = 0.7
>   - Meaning: n_positive_synthetic = 0.7 × n_negative
>   - Formula: 0.7 × 41,496 = 29,047 synthetic positive samples
> - random_state = 42 (reproducible)
> - k_neighbors = 5 (default, to create synthetic samples)
>
> **After SMOTENC:**
> - Negative: 41,496 (original)
> - Positive: 3,826 + 25,221 synthetic = 29,047 total
> - Total rows: 70,543
> - New ratio: 59.3% negative, 40.7% positive
>
> **Critical: SMOTENC applied ONLY to training set**
> - Test set remains 11,331 original (no synthetic samples)
> - Why? Test must represent real-world distribution
>
> **Limitation:** SMOTENC applied before CV, not inside fold
> - Ideal: Inside each CV fold (prevent leakage)
> - Current: Once on full training data
> - Impact: May slightly overestimate performance"

**Sẵn sàng:**
- "Why SMOTENC not undersampling?" → Huy's choice; undersampling loses info
- "Why sampling_strategy 0.7, not 1.0?" → Balance imbalance without 1:1 ratio
- "Does synthetic harm model?" → Slight risk; in ideally CV-wrapped
- "What if we skip SMOTENC?" → Model sees 91.6% negative, biased to negative

### PHẦN 7: MODEL TRAINING (5:15–6:15)

**Kịch bản:**

> "We train CatBoost (Gradient Boosting on Decision Trees) with Huy's hyperparameters.
>
> **Hyperparameters:**
> - iterations: 500 (number of trees)
> - depth: 4 (tree depth, shallow to prevent overfitting)
> - learning_rate: 0.0185
> - l2_leaf_reg: 0.0032 (L2 regularization)
> - random_strength: 0.486 (add randomness to splits)
> - bagging_temperature: 0.874 (control sampling)
> - random_seed: 42
>
> **Class weights:**
> - class_weights = {0: 1, 1: 10}
> - Meaning: Penalize false negatives 10x harder than false positives
> - Why? Prefer recall (catch positives) over precision (avoid false alarms)
>
> **Cross-validation strategy:**
> - 6-fold StratifiedKFold
> - shuffle=True
> - random_state=42
> - Ensures positive rate same in each fold
>
> **Threshold optimization:**
> - For each fold:
>   - Train on 5 folds
>   - Find threshold maximizing F1 or PR-AUC on holdout fold
> - Average thresholds across 6 folds
> - **Final threshold: 0.8564852152742759**
>
> **Why 0.8564852, not 0.5?**
> - Default 0.5 assumes balanced costs
> - Class weights + SMOTENC shift decision boundary
> - High threshold means: predict 1 only when very confident
> - Result: Better precision, lower recall (but consistent with class weights)"

**Sẵn sàng:**
- "Why CatBoost?" → Huy chose it; tree-based, handles categoricals natively
- "What about regularization overfitting?" → depth=4 shallow, l2_leaf_reg penalizes
- "Can threshold change?" → Currently frozen, documented in metadata
- "What if we use threshold 0.5?" → Different TP/FP trade-off, lower recall

### PHẦN 8: EVALUATION (6:15–7:00)

**Kịch bản:**

> "Test set evaluation with frozen model and threshold.
>
> **Confusion matrix (11,331 test samples):**
> ```
>            Predicted 0  Predicted 1
> Actual 0:  10,115 TN        260 FP
> Actual 1:     907 FN         49 TP
>
> Totals:    11,022           309
> ```
>
> **Interpretation:**
> - True Negative (10,115): Predicted no readmission, correct
> - False Positive (260): Predicted readmission, but no readmission occurred
> - False Negative (907): Did NOT predict readmission, but readmission occurred
> - True Positive (49): Predicted readmission, correct
>
> **Metrics:**
>
> | Metric | Formula | Value | Status |
> | --- | --- | --- | --- |
> | Precision | TP/(TP+FP) | 49/309 = 0.159 | Low; many false alarms |
> | Recall | TP/(TP+FN) | 49/956 = 0.051 | **Very low; misses 90%** |
> | F1 | 2×(P×R)/(P+R) | 0.077 | Very low |
> | Specificity | TN/(TN+FP) | 10115/10375 = 0.975 | High; good at finding negatives |
> | FPR | FP/(FP+TN) | 260/10375 = 0.025 | Low (acceptable) |
> | ROC-AUC | Ranking metric | 0.567 | Barely > random (0.5) |
> | PR-AUC | Area under precision-recall curve | 0.108 | Very low for imbalanced data |
> | Brier Score | Mean((pred-actual)²) | 0.386 | High; not well-calibrated |
>
> **Key finding: Recall 5.13%**
> - Model detects only 49 of 956 true readmissions
> - Misses 907 cases (90%)
> - **Not suitable for clinical screening**
>
> **Conclusion:**
> - Model performance is WEAK
> - Engineering is GOOD (reproducible, tested, monitored)
> - Project is DEMO, not production system"

**Sẵn sàng:**
- "Why recall so low?" → High threshold (0.8564852), class_weights, data distribution
- "Can we improve recall?" → Yes, lower threshold; but increases FP
- "Is model useless?" → Clinically yes; but MLOps demo excellent
- "Compared to baseline?" → Random classifier ROC-AUC 0.5; we're 0.567 (barely better)

### PHẦN 9: DEPLOYMENT & SERVING (7:00–8:00)

**Kịch bản:**

> "Now we serve this frozen model as a production system.
>
> **Model artifact bundle** (models/production_huy/):
> - model.pkl: CatBoost classifier (160 MB)
> - preprocessing_state.json: Frozen scaler state (means, scales, min, max)
> - feature_manifest.json: 40 and 52 feature contracts, categorical indices
> - metadata.json: Version, threshold, metrics, training protocol
> - reference_predictions.json: Golden test cases (regression test)
> - reports/: Evaluation metrics, calibration curve, SHAP importance, fairness audit
>
> **Model identity verification:**
> - SHA-256 checksum: a0d11b7ed0c1956d10afbfda360ec24ae2c55f6d6d50d32ed50780a81160331b
> - Verified at:
>   - CI (GitHub Actions)
>   - Container build time
>   - API startup (lifespan)
>   - Prediction time (caching)
>
> **Container (Docker):**
> - Base image: python:3.11.15-slim-bookworm (small)
> - Non-root user (security)
> - Healthcheck: probes /ready endpoint every 30 seconds
> - CMD: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
> - **Does NOT train model**—serves frozen pickle
>
> **API endpoints:**
> - GET /health → Always 200 (is app alive?)
> - GET /ready → 200 or 503 (is model usable?)
> - POST /predict → Prediction with 40 input fields
>
> **Prediction flow:**
> 1. Client sends JSON (40 fields, no patient ID)
> 2. Pydantic validation (schema, bounds, enums)
> 3. build_huy_features() transform 40 → 52
> 4. model.predict_proba()
> 5. risk_score = P(readmission <30)
> 6. if risk_score ≥ 0.8564852 → prediction = 1, else 0
> 7. Return JSON (model_version, risk_score, threshold, prediction, status)
>
> **Monitoring (Prometheus):**
> - HTTP request count/errors/latency
> - Model prediction distribution
> - Model readiness (gauge)
> - Alerts: API unavailable, model not ready, error rate > 5%, latency > 1s
>
> **Dashboard (Grafana):**
> - Request volume (req/sec)
> - P50/P95 latency (ms)
> - Error rate (%)
> - Prediction risk score histogram
> - Model readiness status"

**Sẵn sàng:**
- "Can model update?" → Yes, redeploy with new SHA, all checks auto-validate
- "What if /ready fails?" → /health still 200; /predict returns 503
- "Are predictions logged?" → No (privacy). Only aggregated metrics.
- "Latency requirements?" → P95 < 1s (configured in alerts)

### PHẦN 10: TESTING & CI/CD (8:00–8:45)

**Kịch bản:**

> "All code changes protected by automated tests and CI pipeline.
>
> **Production contract tests:**
>
> Test 1: Feature contract consistency
> ```
> Assert len(REQUEST_FEATURES) == 40
> Assert len(MODEL_INPUT_FEATURES) == 52
> Assert len(CATEGORICAL_MODEL_FEATURES) == 7
> Assert CatBoost.feature_names_ == MODEL_INPUT_FEATURES (exact order)
> Assert CatBoost.get_cat_feature_indices() maps correctly
> ```
>
> Test 2: Model identity frozen
> ```
> Assert model_version == 'huy-catboost-1.0.0'
> Assert decision_threshold == 0.8564852152742759 (exact)
> Assert model_sha256 == 'a0d11b7ed0c1956d10afbfda360ec24ae2c55f6d6d50d32ed50780a81160331b'
> Assert class_weights == {0: 1, 1: 10}
> ```
>
> Test 3: Reference predictions (regression)
> ```
> low_risk = predict(sample_request.json)
> Assert low_risk.risk_score ≈ 0.6643320154788986 (±1e-12)
> Assert low_risk.prediction == 0
>
> high_risk = predict(sample_high_risk_request.json)
> Assert high_risk.risk_score ≈ 0.9379868301418867 (±1e-12)
> Assert high_risk.prediction == 1
>
> # Clear cache, reload, verify exact reproducibility
> load_production_artifacts.cache_clear()
> reloaded = predict(sample_request.json)
> Assert reloaded.risk_score ≈ low_risk.risk_score (±1e-12)
> ```
>
> **CI pipeline (.github/workflows/ci.yml):**
> 1. Verify artifacts exist (model.pkl, preprocessing_state.json, etc.)
> 2. Install dependencies + check compatibility
> 3. Lint & format (ruff check, ruff format --check)
> 4. Run tests (pytest --cov=src --cov-fail-under=60)
> 5. Build Docker image
> 6. Fail if any step fails
>
> **Coverage:** Minimum 60% code coverage required
>
> **Cost of CI:**
> - Time: 2-3 minutes per PR
> - Catches: Regression, contract breaks, format violations
> - Prevents: Shipping broken model, unformatted code, low quality"

**Sẵn sàng:**
- "Why exact tolerance 1e-12?" → IEEE 754 float precision for reproducibility proof
- "What if test fails?" → PR blocked, must fix before merge
- "Can we skip tests?" → No (branch protection + CI required)
- "Is 60% coverage enough?" → For demo, yes; production should 80%+

### PHẦN 11: RESPONSIBLE AI & LIMITATIONS (8:45–9:30)

**Kịch bản:**

> "We acknowledge model limitations explicitly.
>
> **Model performance limitations:**
>
> 1. **Recall 5.13%**: Misses 90% of positive cases
>    - Impact: Not suitable for clinical screening
>    - Implication: Cannot autonomously flag patients at risk
>
> 2. **Precision 15.86%**: 84% of flagged cases are false alarms
>    - Impact: Clinician must review every prediction
>    - False alarm fatigue
>
> 3. **ROC-AUC 0.567**: Barely better than random
>    - Implication: Model has minimal discriminative ability
>    - Potential bias in decision making
>
> 4. **Probability NOT calibrated**: Brier score 0.386 (high)
>    - Raw risk_score should NOT be interpreted as true probability
>    - 0.856 risk_score ≠ 85.6% actual risk
>
> **Data methodology limitations:**
>
> 5. **Preprocessing statistics fit before split**
>    - Scalers fitted on full 56,653 samples
>    - Then split train/test
>    - Results in evaluation leakage
>    - Best practice: Fit inside CV loop per fold
>
> 6. **SMOTENC before CV**
>    - Synthetic samples created once
>    - Then distributed across CV folds
>    - May leak information from fold holdouts
>    - Best practice: Inside each CV fold
>
> 7. **Threshold optimized on training folds**
>    - Likely uses predictions from same fold
>    - Proper: Separate validation set
>    - Impact: Threshold may overfit
>
> **Cohort & feature limitations:**
>
> 8. **First encounter only**
>    - Patient's full history ignored
>    - Potential signal loss
>
> 9. **Medication direction collapsed**
>    - Up, Down, Steady → all become 1 (medication used)
>    - Loses directional information
>
> **Fairness audit:**
>
> - Analyzed race, gender, age subgroups
> - Some disparities observed (documented in reports/)
> - Report is DESCRIPTIVE (show disparities)
> - NOT prescriptive (don't claim fairness achieved)
> - Small groups flagged (n < 200 or positives < 30)
>
> **Responsible use policy:**
>
> - Score is NOT diagnosis
> - Never deny care based on model output
> - Clinician always reviews before action
> - Provide fallback when service unavailable
> - No patient/encounter identifiers stored
> - Prometheus metrics low-cardinality (no PII)"

**Sẵn sàng:**
- "Is model harmful?" → No (demo only); but if deployed, yes without careful review
- "Should we hide limitations?" → No; transparency builds trust
- "Can we improve fairness?" → Yes, by retraining with fairness constraints
- "Will model be used clinically?" → Not from this project; maybe inspire future work

### PHẦN 12: CONCLUSION & QUESTIONS (9:30–10:00)

**Kịch bản:**

> "Let me summarize.
>
> **This project demonstrates:**
> - How to reproduce a model exactly (frozen pickle, contract validation, regression tests)
> - How to serve ML in production (Docker, FastAPI, health/ready checks)
> - How to monitor online (Prometheus, Grafana, alerts, low-cardinality metrics)
> - How to test continuously (CI/CD, artifact verification, coverage)
> - How to be transparent about limitations (model card, responsible AI policy)
>
> **Model performance:**
> - Recall 5.13%, ROC-AUC 0.567 → NOT suitable clinical production
> - But that's expected for course demo
> - Future work: Retrain with proper leakage-free pipeline
>
> **Reproducibility:**
> - Reference predictions exact to ±1e-12 after reload
> - Contract-driven architecture prevents drift
> - CI/CD enforces artifact integrity
>
> **Key insight:**
> - Good engineering ≠ good model performance
> - This project: Good engineering + weak performance (intentional for learning)
>
> Questions?"

---

## PHẦN B: DEFENSIVE ANSWERS (Prepared untuk difficult questions)

### Category 1: Data & Cohort

**Q: "Why did you filter so aggressively (101k → 56k)?"**
A: "Huy's notebook required complete diagnoses (diag_1/2/3), valid demographic (race, gender), and valid discharge disposition to ensure model input consistency. First-encounter policy prevents information leakage. These are standard cohort criteria for healthcare ML."

**Q: "Didn't you lose signal by filtering?"**
A: "Yes, but leakage is worse than signal loss. The 44% filtered are mostly bad data (missing diagnoses) or repeated patient encounters. Keeping them would create data leakage—test set performance wouldn't reflect real-world deployment."

**Q: "Why first encounter, not most recent encounter?"**
A: "Huy's notebook used drop_duplicates(keep='first'); I reproduced that exactly. Ideally we'd verify temporal ordering, but code doesn't sort by date. This is documented as potential limitation."

**Q: "Could recent encounter be more predictive?"**
A: "Yes, potentially. But reproduction is the goal here. We could propose 'v2: most-recent retraining' as future work."

### Category 2: SMOTENC & Class Imbalance

**Q: "Why not balance to 50-50?"**
A: "sampling_strategy=0.7 creates 40.7% positive, 59.3% negative—still imbalanced but not extreme. This was Huy's choice. 50-50 SMOTE might over-represent minority and bias model."

**Q: "Doesn't SMOTENC create fake data?"**
A: "Yes. SMOTE interpolates between existing positive samples in feature space. This helps training, but shouldn't leak to test. SMOTENC applied only to training set; test is original 11,331."

**Q: "Could SMOTENC hurt model?"**
A: "Potential risks: (1) synthetic samples may not reflect true distribution, (2) SMOTENC before CV may leak. Best practice: inside CV per fold. This is documented limitation."

**Q: "Why sampling_strategy=0.7, not balanced?"**
A: "I don't know Huy's exact rationale. Likely: 1:1 ratio too aggressive; 0.7 balance without 100% over-sampling. Common practice in imbalanced learning."

### Category 3: Threshold & Performance

**Q: "Recall 5% is terrible—is model unusable?"**
A: "For clinical screening, yes. But project goal is reproducibility & MLOps demo, not performance. Model demonstrates how to serve ML correctly even with weak performance."

**Q: "Why threshold 0.8564852 instead of 0.5?"**
A: "Optimized via 6-fold CV. High threshold means 'predict 1 only when very confident.' With class_weights={0:1, 1:10}, model penalizes false negatives, but threshold remains high. Different threshold would trade recall vs. precision."

**Q: "Could lower threshold improve recall?"**
A: "Yes. But Huy chose 0.8564852; we reproduce it. Threshold selection is frozen in metadata."

**Q: "Is model biased?"**
A: "Possible. Fairness audit shows disparities (race, gender, age), documented in reports/. Report is descriptive—disparities noted, fairness not claimed."

**Q: "Why not post-hoc calibration?"**
A: "Huy's notebook didn't include it. Probability remains uncalibrated (Brier 0.386). Future v2 could add temperature scaling or isotonic regression."

### Category 4: Data Leakage

**Q: "Isn't preprocessing fit before split leakage?"**
A: "Yes, technically. Scaler fitted on cohort (56,653), then split. Proper approach: fit per fold inside CV loop. This is documented limitation. But for reproducibility goal, frozen state must match notebook."

**Q: "Does leakage invalidate reproducibility?"**
A: "No. Leakage would invalidate REAL-WORLD performance claims. But reproducibility = matching notebook exactly, including leakage. v2 should fix this."

**Q: "Is model performance overestimated?"**
A: "Likely yes, due to leakage. True performance (if tested on separate population) might be worse. Caveat included in MODEL_CARD.md."

### Category 5: Feature Engineering

**Q: "Why 40→52 features? Seems arbitrary."**
A: "Huy's notebook design. Encoding, diagnosis grouping, interactions, log transforms, scaling. We reproduce exactly. If different features needed, would require retraining."

**Q: "Which features most important?"**
A: "SHAP global importance in reports/global_shap_huy_features.csv. Top features (informally): num_medications, num_lab_procedures, age, time_in_hospital. SHAP describes model behavior, not causality."

**Q: "What if feature order wrong?"**
A: "CatBoost fails—embedded feature_names_ validation. CI tests this; cannot merge if contract broken."

**Q: "Can we drop low-importance features?"**
A: "Would require retraining and re-validation. Current project frozen; possible future work."

### Category 6: API & Deployment

**Q: "Why no patient ID in request?"**
A: "Privacy by design. API intentionally forbids identifiers. Score is decision-support, not diagnosis."

**Q: "Can model update?"**
A: "Yes, but requires new SHA256, new metadata, new validation. CI will verify new bundle before merge."

**Q: "What if model.pkl corrupted?"**
A: "SHA256 mismatch detected at startup. /ready returns 503; /predict returns 503. System stays alive but unserving."

**Q: "Are predictions logged?"**
A: "No. Requests not logged. Only Prometheus metrics aggregated (counts, histograms). No individual predictions stored."

### Category 7: Testing & CI

**Q: "Why reference predictions exact to ±1e-12?"**
A: "Floating-point precision test. IEEE 754 double precision; ±1e-12 proves exact reproducibility. Catches numerical drift."

**Q: "What if test fails?"**
A: "PR blocked. Cannot merge. Must fix before resubmit."

**Q: "Is 60% coverage enough?"**
A: "For course demo, yes. Production would need 80%+. Current threshold enforced in CI."

**Q: "Are all code paths tested?"**
A: "Majority of happy path + error paths. Edge cases (malformed JSON, missing fields) covered by Pydantic validation tests."

### Category 8: Monitoring & Alerts

**Q: "Can alerts false-positive?"**
A: "Yes. P95 latency > 1s for 5m can trigger even during peak load. Should tune thresholds post-deployment."

**Q: "How long does detection take?"**
A: "Alert rule 'for: 5m' means alert fires after 5 consecutive evaluation intervals (each 5s). So ~5 minutes minimum."

**Q: "Is Prometheus data persistent?"**
A: "Default: 15-day retention. Data lost after. MLflow stores artifacts permanently (model versions)."

**Q: "Do metrics store patient data?"**
A: "No. Labels: route, status_code, model_version, binary prediction (0/1). No IDs, no risk scores."

### Category 9: Model Card & Responsible AI

**Q: "Why publish limitations?"**
A: "Transparency. Consumers deserve to know weaknesses. Hiding limitations risks misuse."

**Q: "Is model suitable for clinical use?"**
A: "No. Recall 5%, uncalibrated probability, data leakage. This is decision-support demo, not clinical system."

**Q: "What should clinician do?"**
A: "Never automate decision from this model. Use as triage suggestion only. Review patient context. Provide non-model fallback."

**Q: "Can fairness disparities be excused?"**
A: "No. Documented disparities should trigger retraining with fairness constraints or stratified evaluation. Report is descriptive; action required."

### Category 10: Project Scope & Future Work

**Q: "Is this production-ready?"**
A: "MLOps infrastructure: yes. Model performance: no. Safe to deploy in sandbox; not clinical production."

**Q: "What's next?"**
A: "Retraining without leakage, post-hoc calibration, fairness-aware resampling, external validation. Propose as v2."

**Q: "Why not fix now?"**
A: "Frozen Huy model is v1 checkpoint. Better to preserve for learning, propose improvements separately."

**Q: "Can this scale?"**
A: "API can handle reasonable traffic (currently tested ~0.1 req/s). Scaling: add load balancer, replicate containers."

---

## PHẦN C: CHEAT SHEET EXPANDED

### Must-Know Numbers

```
Data:
  101,766 raw encounters
  71,518 unique patients
  56,653 final cohort (-44%)
  45,322 train (no SMOTE)
  70,543 train (after SMOTE)
  11,331 test

Imbalance:
  Train raw: 41,496 neg / 3,826 pos (91.6% / 8.4%)
  Train SMOTE: 41,496 neg / 29,047 pos (59.3% / 40.7%)
  Test: 10,375 neg / 956 pos (91.5% / 8.5%)

Metrics:
  Precision: 0.1586 (15.86%)
  Recall: 0.0513 (5.13%)
  F1: 0.0775
  ROC-AUC: 0.5668
  PR-AUC: 0.1081
  Brier: 0.3856

Confusion:
  TP: 49, FP: 260, FN: 907, TN: 10,115

Features:
  40 request fields
  52 model fields
  7 categorical
  20 medications
  9 interactions

Model:
  CatBoost iterations: 500, depth: 4, lr: 0.0185
  Class weights: {0:1, 1:10}
  Threshold: 0.8564852152742759

Reference:
  Low risk: 0.6643 → 0
  High risk: 0.9379 → 1

SHA256: a0d11b7ed0c1956d10afbfda360ec24ae2c55f6d6d50d32ed50780a81160331b
Version: huy-catboost-1.0.0
```

### Call-Out Facts (能快速答出的)

- "101,766 to 56,653 because of filters + first encounter policy"
- "70,543 because SMOTENC: 41,496 + 29,047 synthetic"
- "Recall 5.13% means model misses 90% of readmissions"
- "Threshold 0.8564852 optimized via 6-fold CV, not default 0.5"
- "Preprocessing state frozen, not refitted—data leakage documented"
- "Reference predictions exact to ±1e-12—reproducibility proof"
- "Model performance weak, MLOps infrastructure strong"

### Dangerous Wrong Answers (絕對避免)

❌ "SMOTE applied to test set too"
✅ "SMOTE only on training, test original"

❌ "Threshold 0.5 default"
✅ "Optimized 0.8564852"

❌ "No data leakage"
✅ "Preprocessing fit before split, documented"

❌ "Model suitable clinical production"
✅ "Demo only, recall 5%"

❌ "Probability is calibrated"
✅ "Uncalibrated, Brier 0.386"

❌ "40 features same as 52"
✅ "40 raw, 52 engineered (40→52 transform)"

❌ "SMOTENC inside CV"
✅ "Before CV, potential leakage"

❌ "Model trains on container start"
✅ "Serves frozen model.pkl"

---

## PHẦN D: 1-HOUR EMERGENCY STUDY PLAN

**If you have 1 hour to learn before presentation:**

### 0:00–0:05 (5 min): Core Numbers
- Write down: 101k, 56k, 45k, 70k, 11k
- Write down: Recall 5%, ROC 0.567, threshold 0.8564852
- Write down: 40→52 features, 7 categorical

### 0:05–0:15 (10 min): Architecture Diagrams
- Draw: Raw data → Cohort → Split → SMOTE → CatBoost → Threshold → Test
- Draw: API startup (load model → validate → ready)
- Draw: Prediction flow (40 fields → 52 features → score → threshold → response)

### 0:15–0:25 (10 min): Prepare 3 Key Stories
1. "101k → 56k because..." (filters + first encounter)
2. "70k because..." (SMOTENC before/after)
3. "Recall 5% means..." (misses 90%, not suitable clinical)

### 0:25–0:35 (10 min): Understand Limitations
- Data leakage: preprocessing fit before split
- SMOTENC: before CV, not inside fold
- Model: weak performance (recall 5%, ROC 0.567)
- Probability: uncalibrated (Brier 0.386)

### 0:35–0:45 (10 min): Practice Opening (2 min)
- "Problem: Predict 30-day readmission. Our goal: reproduce Huy's model + serve it production-ready."
- Deliver with confidence 2x.

### 0:45–0:55 (10 min): Prepare Defense
- "Why 101k → 56k?" Answer memorized.
- "Recall 5% useless?" Answer: "Demo, not clinical."
- "Data leakage?" Answer: "Yes, documented."
- Rehearse 5 hardest Q&A.

### 0:55–1:00 (5 min): Calm & Breathe
- Confidence check
- No last-minute cramming
- Trust your prep

---

## QUICK REFERENCE: "I FORGOT" RECOVERY PHRASES

If you forget a number, use recovery phrase:

**"I don't recall exact number, but conceptually..."**
- "...raw 100k → final 55-60k due to filters"
- "...train ~45k, test ~11k, SMOTE brings train to ~70k"
- "...recall single digit percent" (instead of exact 5.13%)

**"The exact threshold I have here..."**
- (Show metadata.json or cheat sheet)

**"I should double-check this specific claim..."**
- (Suggest looking at MODEL_CARD.md)

**"That's an interesting question. The code shows..."**
- (Redirect to code, shows confidence)

Never make up a number. Say "I'm not certain" → credibility preserved.

---

END OF PRESENTATION & DEFENSE COMPANION
