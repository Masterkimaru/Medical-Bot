#!/bin/bash
# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data to persistent storage
python -c "
import nltk
import os
from pathlib import Path

path = '/opt/render/nltk_data'
Path(path).mkdir(parents=True, exist_ok=True)
nltk.download('punkt_tab', download_dir=path)
nltk.download('punkt', download_dir=path)
nltk.download('wordnet', download_dir=path)
nltk.download('stopwords', download_dir=path)
"