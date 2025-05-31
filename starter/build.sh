#!/bin/bash

# Make sure MODEL_DIR exists
mkdir -p starter/model

# Clean and preprocess data
python starter/data/clean_data.py

# Train model
cd starter && python starter/train_model.py && cd ..
