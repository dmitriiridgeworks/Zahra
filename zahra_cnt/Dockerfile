# Use official Python image
FROM python:3.14-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file from zahra_cnt
COPY zahra_cnt/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# App code will be mounted via docker-compose (no COPY here)

# Set working directory to app code
WORKDIR /app/zahra

# Run the application
CMD ["python", "main.py"]

#cd zahra_cnt
#docker-compose build
#docker-compose run --rm zahra
