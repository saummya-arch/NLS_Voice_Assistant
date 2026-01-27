from datetime import datetime

import streamlit as st
import sounddevice as sd
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

from va_llm import ExtractorLLM, ModifyCalenderLLM, ReplyLLM
from tts import TTS

import requests
import time

from chat_history import ChatHistory

st.title("Voice Assistant(Testing)")
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# ASR model
@st.cache_resource
def load_asr():
    model_name = "distil-whisper/distil-medium.en"
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
def load_extractor_llm():
    llm_model = ExtractorLLM()
    return llm_model

@st.cache_resource
def load_modify_llm():
    llm_model = ModifyCalenderLLM()
    return llm_model

@st.cache_resource
def load_reply_llm():
    llm_model = ReplyLLM()
    return llm_model


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ChatHistory()

# Parameters
processor, model = load_asr()
tts_model = load_tts()
extractor_llm_model = load_extractor_llm()
modify_calender_llm_model = load_modify_llm()
reply_llm_model = load_reply_llm()
duration = 15
sample_rate = 16000

# URL
weather_url = "https://api.responsible-nlp.net/weather.php"
calender_url = "https://api.responsible-nlp.net/calendar.php"


def record_audio(sr=sample_rate, silence_limit=3.0, threshold=0.02, min_duration=3.0):
    st.write("Recording...")
    recorded_chunks = []
    silent_chunks = 0
    total_chunks = 0

    #chunk parameters
    chunk_size = 1024
    limit_in_chunks = int(silence_limit * sr / chunk_size)
    min_chunks = int(min_duration * sr / chunk_size)

    def callback(indata, frames, time, status):
        nonlocal silent_chunks, total_chunks
        total_chunks += 1
        
        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata)) #RMS calculation-eucli norm
        recorded_chunks.append(indata.copy())
        
        #check only after min duration
        if total_chunks > min_chunks:
            if volume_norm < threshold: #chunk is silent
                silent_chunks += 1
            else:
                silent_chunks = 0  # not silent
    print("sr:",sr)
    with sd.InputStream(samplerate=sr, channels=1, callback=callback, blocksize=chunk_size):
        while silent_chunks < limit_in_chunks:
            sd.sleep(100)  # Check status every 100ms

    st.write("Recording over...")
    return np.concatenate(recorded_chunks).flatten()


def predict_audio(audio, sr=sample_rate):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**inputs)
    res = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return res


def weather_process(text, result):
    #handle weather
    weather = result.weather
    city = weather.city if weather.city else "Marburg"

    try:
        response = requests.post(weather_url, data={'place': city}).json()
        print(response)
        weekday = weather.day if weather.day else datetime.now()
        dt = datetime.strptime(weekday, "%Y-%m-%d")

        wd = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        weekday = wd[dt.weekday()]
        res = next(forecast for forecast in response['forecast'] if forecast['day']==weekday)
        temp = res['temperature']
        weather_desc = res['weather']

        response = reply_llm_model.chat(text, {'city': city, 'temp': temp, 'weather': weather_desc})
        print("weather response",response)
        if response.answer:
            response_fin = response.answer.replace("**", "")

        return response_fin, {'city': weather.city}
    
    except Exception as e:
        print(e)
        return 'Error on accesing weather information.'
    
def calender_process(text, result):
    calender_param = {"calenderid": "uid23"}
    entries = requests.get(calender_url, params=calender_param).json()
    print(entries)
    response = 'Failed to process calender request, please try again.'
    calender = result.calender
    response = modify_calender_llm_model.chat(text, entries)
    print('llm output for calender:', response)
    if not calender.create_new_entry:
        if response:
            print('in get', ids := response.ids)
            if response.request == 'get':
                for entry in entries:
                    if entry['id'] in ids:
                        response = f'Event is {entry["title"]} for {entry["description"]}.'
            elif response.request == 'delete':
                for entry in entries:
                    if entry['id'] in ids:
                        requests.delete(calender_url, params = {**calender_param, 'id': entry['id']})
                        response = 'Event is deleted.'
            elif response.request == 'put':
                event_data = response.calender.model_dump()
                response = requests.put(calender_url, params={**calender_param, **event_data})
        else:
            response = 'Failed to process calender request, please try again.'
    elif calender.create_new_entry:
        event_data = {
            "title": calender.title,
            "description": calender.description,
            "start_time": calender.start_time,
            "end_time": calender.end_time,
            "location": calender.location,
        }
        response = requests.post(calender_url, params = {**calender_param, **event_data}).json()
        print('response for create', response)
        if response.status_code == 200:
            response = 'Event has been set'
        else:
            response = 'Failed to set event'
    print("calender response:",response)
    return response, {}

def voice_assistant():

    chat_history = st.session_state.chat_history

    if chat_history.chats:
        st.subheader("Chat history")
        for chat in chat_history.chats[-4:]:
            st.text(f"You:{chat.usr_request}")
            st.text(f"Bot:{chat.chat_response}")
            st.divider()

    if st.button("Record Audio"):

        # record audio
        audio = record_audio()
        with st.spinner("Transcribing!!"):
            text = predict_audio(audio)

        print(text)
        st.text(text)

        # LLM testing
        result = extractor_llm_model.chat(text)
        print("LLm response:",result)

        if result.intent == 'calender':
            response, entries = calender_process(text, result)
        elif result.intent == 'weather':
            response, entries = weather_process(text, result)

        chat_history.add(
            usr_request=text,
            chat_response=response,
            intent=result.intent,
            entries=entries,
        )
        start_time = time.time()

        # tts the response
        audio = tts_model.speak(response)
        print('Total tts infer time:', time.time() - start_time)

        # rerun record
        st.rerun()
    
    if st.button("Clear History"):
        chat_history.clear()
        st.rerun()

# try:
#     voice_assistant()
# except Exception as e:
#     print(e)
#     st.rerun()

voice_assistant()