import streamlit as st
import sounddevice as sd
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

from va_llm import LLM
from tts import TTS

import requests
import spacy

import time

st.title("Voice Assistant(Testing)")
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# ASR model
@st.cache_resource
def load_asr():
    model_name = "distil-whisper/distil-small.en"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to("cpu")
    model.eval()
    return processor, model


# TTS model
@st.cache_resource
def load_tts():
    tts_model = TTS()
    return tts_model


@st.cache_resource
def load_llm():
    llm_model = LLM()
    return llm_model


# Parameters
processor, model = load_asr()
tts_model = load_tts()
llm_model = load_llm()
duration = 15
sample_rate = 16000
api_key = ""
keywords = ['current', 'today', 'now', 'tomorrow', 'yesterday']
nlp = spacy.load("en_core_web_sm")


def record_audio(seconds=duration, sr=sample_rate):
    st.write("Recording...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    # audio = sd.rec(samplerate=sr, channels=1, dtype="float32")
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

def voice_assistant():
    if st.button("Record Audio"):

        # record audio
        audio = record_audio(duration)
        with st.spinner("Transcribing!!"):
            text = predict_audio(audio)

        print(text)
        st.text(text)

        # LLM testing
        result = llm_model.chat(text)
        print(result)

        if result.intent == 'calender':
            response = 'Failed to process calender request'
            calender = result.calender
            calender_url = "https://api.responsible-nlp.net/calendar.php"
            calender_param = {"calenderid": "uid23"}
            if calender.intent == 'fetch_data':
                response = requests.get(calender_url, params=calender_param).json()
                print('fetched', response)
                if response:
                    for entry in response:
                        tts_model.speak(
                            f'Title: {entry["title"]} Description: {entry["description"]} Time: {entry["start_time"]}')
                        response = 'Those are all the meetings'
                else:
                    response = 'Failed to set meeting'
            elif calender.intent == 'update_data':
                meeting_data = {
                    "title": calender.title,
                    "description": calender.description,
                    "start_time": calender.start_time,
                    "end_time": calender.end_time,
                    "location": calender.location,
                }
                response = requests.post(calender_url, params=calender_param, json=meeting_data)
                if response[0] == 200:
                    response = 'Meeting set'
                else:
                    response = 'Failed to set meeting'
            print(response)

        start_time = time.time()

        # tts the response
        audio = tts_model.speak(response)
        print('Total tts infer time:', time.time() - start_time)

        # rerun record
        st.rerun()


voice_assistant()
