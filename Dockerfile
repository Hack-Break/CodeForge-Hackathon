# ────────────────────────────────────────────────────────────────
# NeuralPath — Multi-stage Dockerfile
# Stage 1: Build React frontend
# Stage 2: Python backend + serve built frontend
# ────────────────────────────────────────────────────────────────

# ── Stage 1: React frontend ──────────────────────────────────
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install --silent
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend ───────────────────────────────────
FROM python:3.11-slim AS backend

# System deps: gcc for C extensions (spaCy tokeniser), poppler for PDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download spaCy model so first request is instant
RUN python -m spacy download en_core_web_sm

# Pre-cache BERT NER model weights (requires internet at build time).
# Written as a script file to avoid shell quoting issues on all platforms.
RUN printf 'try:\n\
    from transformers import pipeline\n\
    pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple", device=-1)\n\
    print("BERT NER model cached.")\n\
except Exception as e:\n\
    print(f"BERT pre-cache skipped ({e}) - will download on first request.")\n\
' > /tmp/cache_bert.py && python /tmp/cache_bert.py

# Copy backend source
COPY backend/ ./backend/

# Copy evaluation scripts and create data dir for dataset uploads
COPY scripts/ ./scripts/
RUN mkdir -p /app/data

# Copy compiled React frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]