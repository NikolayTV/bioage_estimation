# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY models/ models
COPY fastapi_bioage_prediction.py fastapi_bioage_prediction.py
COPY avgs_for_age_group.csv avgs_for_age_group.csv

CMD ["uvicorn", "fastapi_bioage_prediction:app", "--host", "0.0.0.0", "--port", "80"]
