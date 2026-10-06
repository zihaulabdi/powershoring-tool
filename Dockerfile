# Container image for the Morocco Powershoring tool.
# Build:  docker build -t powershoring-tool .
# Run:    docker run -p 8501:8501 powershoring-tool   ->  http://localhost:8501
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so code changes do not invalidate this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code, data and Streamlit theme
COPY . .

EXPOSE 8501

# Streamlit's built-in health endpoint, for orchestrators and load balancers
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "How_To.py", "--server.port=8501", "--server.address=0.0.0.0"]
