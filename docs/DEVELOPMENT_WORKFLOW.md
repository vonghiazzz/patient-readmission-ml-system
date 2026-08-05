# Development and Testing Workflow

## 1. Branch strategy

feature/<task-or-role>
        ↓ Pull Request
develop
        ↓ Release Pull Request
main

Không commit trực tiếp vào develop hoặc main.

Các branch chính:

feature/data-pipeline: Data ingestion, validation, splitting, preprocessing
feature/model-training: Model training and evaluation
feature/api: FastAPI and model serving
feature/monitoring: Monitoring and observability
feature/responsible-ai: Responsible AI documentation and evaluation

## 2. Environment setup

Từ project root:

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check

Không commit:

.venv/
.env
raw/interim patient data
serialized local model artifacts
caches


3. Common quality checks

Mọi thành viên phải chạy trước khi tạo Pull Request:

python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -v

Điều kiện:

Ruff check passed
Ruff format check passed
Pytest có 0 failed và 0 errors

Warning phải được đọc và ghi nhận, nhưng chỉ chặn merge nếu ảnh hưởng tính đúng đắn hoặc an toàn.

4. Role-specific testing
Member A — Data
python -m pytest tests/data -v

Local pipeline smoke test:

python -m src.data.ingestion
python -m src.data.validation
python -m src.data.splitting
python -m src.features.preprocessing

Kiểm tra output local:

ls -lh data/interim/ingested_data.csv
ls -lh data/interim/splits/train.csv
ls -lh data/interim/splits/validation.csv
ls -lh data/interim/splits/test.csv
ls -lh models/preprocessor.joblib

Patient-level CSV và .joblib không được commit.

Member B — ML

Khi thư mục test model đã được tạo:

python -m pytest tests/models -v

Training phải sử dụng:

Split do DATA-06 tạo
Preprocessor do DATA-07 tạo
Cấu hình versioned
Fixed random seed
MLflow tracking

Test set không được dùng để tuning hoặc chọn model.

Member C — Backend
python -m pytest tests/integration -v
python -m pytest tests/unit -v

Nếu chưa có tests/unit, chỉ chạy thư mục đang tồn tại.

API smoke test:

python -m uvicorn src.api.main:app \
  --host 127.0.0.1 \
  --port 8000

Terminal khác:

curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
curl -i http://127.0.0.1:8000/openapi.json

Các endpoint trên phải trả HTTP 200.

Member D — MLOps/Docs

Khi Docker Compose đã được triển khai:

docker compose config
docker compose build
docker compose up -d
docker compose ps

Kiểm tra:

curl -i http://localhost:8000/health

Sau test:

docker compose down

Không commit secret, credential, .env, mlruns/ hoặc volume data local.


5. Feature branch to develop

Trước khi tạo PR:

git switch feature/<branch>
git pull --ff-only origin feature/<branch>

python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -v

git status
git push

Tạo Pull Request:

base: develop
compare: feature/<branch>

Chỉ merge khi:

CI passed
Review approved
Không conflict
Acceptance criteria của task đạt
Evidence đã được cập nhật
Không có raw data hoặc binary artifact trong PR


6. Develop to main

Sau khi các feature PR đã merge:

git switch develop
git pull --ff-only origin develop

python -m pip install -r requirements.txt
python -m pip check
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -v

Chạy thêm smoke tests phù hợp:

Data pipeline local smoke test
API health/readiness smoke test
Docker Compose smoke test nếu đã triển khai