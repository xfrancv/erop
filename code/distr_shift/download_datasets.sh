#!/bin/bash
# Download all datasets

source .venv/bin/activate

python download_datasets.py
python analyze_datasets.py