# Use image with TA-Lib pre-installed
FROM python:3.9-slim

WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# Copy application
COPY . .

# Create user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0", "--server.headless=true"]

