#!/bin/bash

# Export the environment variables
source .env

# Make temporary directory for data download
mkdir -p data/images/raw

# Download the data
kaggle competitions download dogs-vs-cats -p data/images/raw

# # Decompress the data
# unzip data/raw/dogs-vs-cats.zip -d data/images/tmp

# # Decompress the training data
# unzip data/raw/train.zip -d data/interim

# # Make target directories for training data
# mkdir data/processed/train
# mkdir data/processed/test
# mkdir data/processed/train/dog
# mkdir data/processed/train/cat

# # Move the training dogs and cats and the testing data
# mv data/interim/train/dog.* data/processed/train/dog/
# mv data/interim/train/cat.* data/processed/train/cat/
# mv data/interim/test1/* data/processed/test/

# # Clean up
# rm data/raw/sampleSubmission.csv