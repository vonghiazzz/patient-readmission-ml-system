# Problem Definition — Patient Readmission Prediction

## 1. Problem Statement
Bệnh viện cần xác định bệnh nhân có nguy cơ tái nhập viện trong vòng 30 ngày
sau khi xuất viện, để ưu tiên các biện pháp theo dõi, tư vấn thuốc và chăm sóc
sau xuất viện.

Hệ thống KHÔNG tự quyết định điều trị. Nó chỉ cung cấp:
- Risk probability
- Risk level (low / medium / high)
- Contributing factors

## 2. Prediction Point
Dự đoán được thực hiện tại hoặc ngay trước thời điểm xuất viện.
Chỉ dùng dữ liệu có sẵn trước/tại thời điểm đó.

Không được dùng:
- Thông tin phát sinh sau khi xuất viện
- Kết quả tái nhập viện thực tế
- Feature liên quan trực tiếp tới tương lai
- Dữ liệu của lần nhập viện tiếp theo

## 3. Target Definition (Diabetes 130-US Hospitals)
- `<30`  → 1 (readmitted within 30 days)
- `>30`  → 0
- `NO`   → 0

> Cần đối chiếu lại với data dictionary thực tế trước khi triển khai chính thức (phối hợp với Member A).

## 4. Users & Use Case
**Người dùng chính:** Care management team, Discharge planning team, Clinical operations team.

Nhân viên nhập thông tin bệnh nhân → Hệ thống trả risk score →
Bệnh nhân được xếp hạng nguy cơ → Nhóm chăm sóc quyết định mức can thiệp.

## 5. Functional Requirements
- Nhận thông tin bệnh nhân qua REST API
- Validate dữ liệu đầu vào
- Trả xác suất tái nhập viện + risk level + top contributing factors
- Health check endpoint
- Ghi nhận metrics (Prometheus)
- Không ghi dữ liệu nhạy cảm vào log

## 6. Non-Functional Requirements
- API p95 latency < 200–300ms
- Model load một lần khi khởi động app
- Validation & error response rõ ràng
- Container có health check
- Không chứa patient ID trong log
- Kết quả tái lập được từ model version + pipeline version
- Có khả năng rollback model

## 7. Success Metrics
**Business (proxy):**
- Trong top 30% bệnh nhân risk cao nhất, hệ thống phát hiện được ≥70% ca readmission thực tế
- Tỷ lệ cảnh báo giả (false positive rate) ở mức chấp nhận được cho capacity đội chăm sóc

**Model:** AUROC, AUPRC, Recall, Precision, F1, Specificity, Brier score, Calibration curve

**System:** p95 latency, error rate, throughput, model loading status, missing-feature rate

## 8. Scope & Out-of-Scope
**In scope:** Prediction service, monitoring, Responsible AI, Docker Compose deployment.
**Out of scope:** Không thay thế quyết định lâm sàng của bác sĩ; không dự đoán các bệnh lý khác ngoài readmission; không xử lý real-time streaming dữ liệu HL7/FHIR.