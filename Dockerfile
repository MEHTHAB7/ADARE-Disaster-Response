# Stage 1: Build React UI
FROM node:18-alpine AS builder
WORKDIR /ui
COPY ui/package*.json ./
RUN npm install
COPY ui/ .
RUN npm run build

# Stage 2: Serve FastAPI Backend
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install key dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Copy UI build
COPY --from=builder /ui/dist /app/ui/dist

EXPOSE 7860

CMD ["python", "server.py"]
