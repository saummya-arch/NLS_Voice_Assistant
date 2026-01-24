# copy paste this command :

set DISTIL_MODEL=distil-whisper/distil-medium.en

# Command to run streamlit :

streamlit run .\tts_calender.py

# Ollama setup - please check the link for instructions

https://github.com/ollama/ollama

# to run in docker use these commands:

docker build -t ms1-distil .
docker run --rm -p 8000:8000 -e DISTIL_MODEL=distil-whisper/distil-medium.en ms1-distil
