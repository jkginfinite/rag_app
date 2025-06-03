#!/bin/bash

# Download model if not already downloaded
MODEL_PATH="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    mkdir -p models
    gdown --id 1d0aNc8SLgCnPFzBfFWZT_kQMALNyUwRV -O "$MODEL_PATH"
fi

# Run the app
python app.py
