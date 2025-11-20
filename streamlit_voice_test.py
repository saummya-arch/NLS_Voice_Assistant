import streamlit as st
import sounddevice as sd
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
from TTS.api import TTS

import requests
import spacy

import time

st.title("Weather Voice Assistant(Testing)")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#ASR model
@st.cache_resource
def load_asr():
    model_name = "distil-whisper/distil-small.en" 
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to("cpu")
    model.eval()
    return processor, model

#TTS model
@st.cache_resource
def load_tts():
    
    # model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    # tts = TTS(model_name).to("cpu")
    # tts_model = TTS(
    #     model_path="tts_v2", 
    #     config_path="tts_v2/config.json", 
    #     progress_bar=True
    # ).to("cpu")

    #coqui tts
    tts_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=True)
    tts_model.eval()
    
    return tts_model

#Parameters
processor, model = load_asr()
tts_model = load_tts()
duration = 5
sample_rate = 16000
api_key = ""
keywords = ['current', 'today', 'now', 'tomorrow', 'yesterday']
nlp = spacy.load("en_core_web_sm")


def record_audio(seconds=duration, sr=sample_rate):
    st.write("Recording...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    st.write("Recording over...")
    return audio.flatten()

def predict_audio(audio, sr=sample_rate):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**inputs)
    res = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return res

def get_keywords(text):
    
    d = {}
    d['city'] = ""
    d['time'] = ""
    doc = nlp(text=text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            print(ent.text)
            d["city"] = ent.text
    for token in doc:
        if token.text in keywords:
            # print(token)
            d['time'] = token.text
    return d


# duration = st.slider("Recording Duration (seconds)", 1, 10, 3)

if st.button("Record Audio"):

    #record audio
    audio = record_audio(duration)
    with st.spinner("Transcribing!!"):
        text = predict_audio(audio)

    print(text)
    st.text(text)

    #extract keywords
    key_dict = get_keywords(text)

    if key_dict['city'] == '':
        location = requests.get("https://ipinfo.io")
        data = location.json()
        city = data.get("city")
    else:
        city = key_dict['city']
    print("city:", city)

    #get weather update
    city_res = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}").json()
    print(city_res)
    if city_res['cod']==200:
        temp = city_res['main']['temp']- 273.15
        temp = f"{temp:.1f})" #f"{(city_res['main']['temp'] - 273.15):.1f}"
        desc = city_res['weather'][0]['description']

        response = f"The current temperature is {temp}°celsius in {city}, And it's going to be {desc} today"

    else:
        response = "Please repeat the question again"

    start_time = time.time()

    #tts the response
    audio = tts_model.tts(response)
    print('Total tts infer time:', time.time()-start_time)
    st.audio(np.array(audio), sample_rate=22500, format="audio/wav", autoplay=True, loop=False)
    
    st.success(text)



