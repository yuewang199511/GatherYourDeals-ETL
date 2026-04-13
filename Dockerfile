FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install git+https://github.com/yuewang199511/GatherYourDeals-SDK.git

COPY . .

EXPOSE ${PORT:-8080}

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
