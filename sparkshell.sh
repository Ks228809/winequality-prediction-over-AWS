#!/bin/bash

# Define the S3 bucket and folder path
S3_BUCKET="s3://aws-logs-992382847407-us-east-1/EMAP/"

# Define the local directory to store files
LOCAL_DIR="/home/hadoop/EMAP_files"

# Create local directory if it doesn't exist
mkdir -p ${LOCAL_DIR}

# Copy all files from the S3 bucket to the local directory
aws s3 cp ${S3_BUCKET} ${LOCAL_DIR} --recursive

# Install Python dependencies
pip install -r ${LOCAL_DIR}/requirements.txt

# Submit the PySpark job
spark-submit --master yarn ${LOCAL_DIR}/model_training.py

