#!/bin/bash

set -e

echo "Copying .env.example to .env..."
cp .env.example .env

echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt

echo "Starting model training..."
python trainer.py train --batch_size=1 --grad_accum_steps=4 --warmup_steps=10 --max_steps=100 --learning_rate=1e-5

echo "Starting the server..."
uvicorn server:app
