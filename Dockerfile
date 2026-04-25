FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for PostgreSQL and building packages
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to speed up future builds
COPY requirements.txt .

# Install all libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . .

# Expose ports; FastAPI: 8000, MLflow: 5000
EXPOSE 8000
EXPOSE 5000