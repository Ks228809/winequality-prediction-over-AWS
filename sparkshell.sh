#!/bin/bash

# Define the S3 bucket and folder path
S3_BUCKET="s3://aws-logs-992382847407-us-east-1/EMAP/"

# Define the local directory to store files
LOCAL_DIR="/home/hadoop/EMAP_files"

# Define the location of the JAR file and class name
JAR_FILE="${LOCAL_DIR}/WineClassification.jar"
MAIN_CLASS="wineClassification.LogisticRegressionPrediction"

# Create local directory if it doesn't exist
mkdir -p ${LOCAL_DIR}

# Copy all files from the S3 bucket to the local directory
aws s3 cp ${S3_BUCKET} ${LOCAL_DIR} --recursive

# Assuming the requirements include Java dependencies or setup
# Here we could set Java environment or download necessary Java libraries

# Run the Java training class
java -cp ${JAR_FILE} ${MAIN_CLASS}
