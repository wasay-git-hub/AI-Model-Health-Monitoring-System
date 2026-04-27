FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 5000

CMD ["uvicorn", "src.fast_api.api_app:app", "--host", "0.0.0.0", "--port", "8000"]