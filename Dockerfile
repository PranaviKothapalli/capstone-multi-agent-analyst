FROM python:3.11-slim

# Native libraries required by WeasyPrint (PDF rendering) — Section 16.3
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpango-1.0-0 libpangocairo-1.0-0 libcairo2 \
    libgdk-pixbuf2.0-0 libffi-dev shared-mime-info \
    && rm -rf /var/lib/apt/lists/*




WORKDIR /app

RUN apt-get update && apt-get install -y curl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p workspace/uploads workspace/cleaned workspace/features \
    workspace/models workspace/visualizations workspace/reports workspace/metadata

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
