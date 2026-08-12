# Problem Definition — 30-Day Readmission

The system estimates relative risk that a diabetes encounter will be followed by readmission within
30 days. The prediction point is at or immediately before discharge.

Target mapping:

```text
<30 → 1
>30 → 0
NO  → 0
```

The service returns model version, raw risk score, decision threshold, binary prediction and readable
status. It does not return a diagnosis or treatment recommendation.

Success is measured with PR-AUC, ROC-AUC, precision, recall, F1, Brier score, calibration evidence,
subgroup metrics and reproducible artifact checks. System measures include readiness, latency, error
rate and throughput.

This course system is not approved for autonomous clinical use.
