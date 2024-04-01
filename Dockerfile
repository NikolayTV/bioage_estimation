# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY models/ models
COPY fastapi_bioage_prediction.py fastapi_bioage_prediction.py
COPY avgs_for_age_group.csv avgs_for_age_group.csv

# Specify the command to run on container start
CMD ["uvicorn", "fastapi_bioage_prediction:app", "--host", "0.0.0.0", "--port", "80"]
