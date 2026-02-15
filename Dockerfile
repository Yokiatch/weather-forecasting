FROM python:3.9-slim

WORKDIR /app

# Install system tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set Python Path so it can find your scripts folder
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Added --server.headless=true to prevent it from trying to open a browser window
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]