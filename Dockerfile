FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements-export.txt .
RUN pip install --no-cache-dir -r requirements-export.txt
COPY export_model.py .
RUN mkdir -p /model && python export_model.py

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=builder /model /model
COPY main.py segment.py recolor.py ./
ENV MODEL_DIR=/model
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
