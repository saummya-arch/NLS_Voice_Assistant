# Run this for distil whisper setup :

hf download distil-whisper/distil-medium.en --local-dir ./models/whisper

# Command to run streamlit :

streamlit run .\tts_calender.py

# Ollama setup - please check the link for instructions

https://github.com/ollama/ollama

# To run in docker use these commands :

docker compose up -d --build
docker exec -it ollama ollama pull mistral
