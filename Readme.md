# copy paste this command :

set DISTIL_MODEL=distil-whisper/distil-medium.en

# Command to run the Python Fast API Serve :

uvicorn app.main:app --host 127.0.0.1 --port 8000

# to run in docker use these commands:

docker build -t ms1-distil .
docker run --rm -p 8000:8000 -e DISTIL_MODEL=distil-whisper/distil-medium.en ms1-distil
