from datetime import datetime

import streamlit as st
import sounddevice as sd
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa

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
    model_name = "./models/whisper"
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
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

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

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


def record_audio(sr=sample_rate, silence_limit=5.0, threshold=0.02, min_duration=3.0):
    st.write("Recording...")
    recorded_chunks = []
    silent_chunks = 0
    total_chunks = 0

    # chunk parameters
    chunk_size = 1024
    limit_in_chunks = int(silence_limit * sr / chunk_size)
    min_chunks = int(min_duration * sr / chunk_size)

    def callback(indata, frames, time, status):
        nonlocal silent_chunks, total_chunks
        total_chunks += 1

        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))  # RMS calculation-eucli norm
        recorded_chunks.append(indata.copy())

        # check only after min duration
        if total_chunks > min_chunks:
            if volume_norm < threshold:  # chunk is silent
                silent_chunks += 1
            else:
                silent_chunks = 0  # not silent

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
    # handle weather
    weather = result.weather
    city = weather.city if weather.city else "Marburg"

    try:
        response = requests.post(weather_url, data={'place': city}).json()
        print(response)
        weekday_str = weather.day if weather.day else datetime.now().strftime("%Y-%m-%d")
        dt = datetime.strptime(weekday_str[:10], "%Y-%m-%d")
        wd = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        weekday = wd[dt.weekday()]
        res = next(forecast for forecast in response['forecast'] if forecast['day'] == weekday)
        temp = res['temperature']
        weather_desc = res['weather']

        response = reply_llm_model.chat(text, {'city': city, 'temp': temp, 'weather': weather_desc})
        print("weather response", response)
        if response.answer:
            response_fin = response.answer.replace("**", "")

        return response_fin, {'city': weather.city}

    except Exception as e:
        print(e)
        return 'Error on accesing weather information.', {}


def format_date(date_str):
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        day = dt.day
        if 11 <= day <= 13:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        return f"{day}{suffix} {dt.strftime('%B')}"
    except:
        return date_str


def format_entry_details(entry):
    parts = ["Event"]
    if entry.get('title'):
        parts.append(f"with title {entry['title']}")
    if entry.get('description') and entry.get('description') != 'No description':
        parts.append(f"description {entry['description']}")
    if entry.get('start_time'):
        readable_date = format_date(entry['start_time'])
        parts.append(f"on {readable_date}")
    if entry.get('location') and entry.get('location') != 'Not specified':
        parts.append(f"in {entry['location']}")
    return ', '.join(parts) if len(parts) > 1 else 'Event'


def calender_process(text, result, chat_history):
    calender_param = {"calenderid": "uid23"}
    entries = requests.get(calender_url, params=calender_param).json()
    print(entries)

    calender = result.calender
    last_calender_id = chat_history.get_last_appointment()  # prev reference
    created_entry_id = None

    if calender.request_type == 'post':
        event_data = {
            "title": calender.title or 'Untitled',
            "description": calender.description or 'No description',
            "start_time": calender.start_time or 'No start time',
            "end_time": calender.end_time or 'No end time',
            "location": calender.location or 'No location',
        }
        print("post event_data:", event_data)
        post_response = requests.post(calender_url, params=calender_param, json=event_data)
        print('post_response', post_response)
        if post_response.status_code == 200:
            details = format_entry_details(event_data)
            response = f'{details} has been created.'
            response_id = post_response.json()
            created_entry_id = response_id["id"]
        else:
            response = 'Failed to create event'
        return response, {'calendar_entry': created_entry_id}

    llm_response = modify_calender_llm_model.chat(text, entries, last_calender_id)

    print('llm output for calender:', llm_response)

    if llm_response:
        print('in get', ids := llm_response.ids)
        if llm_response.request == 'get':
            required_entires = [entry for entry in entries if entry['id'] in ids]
            response = reply_llm_model.chat(text, required_entires, is_calender_event=True)
            if response.answer:
                response = response.answer.replace("**", "")
        elif llm_response.request == 'delete':
            for entry in entries:
                if entry['id'] in ids:
                    del_response = requests.delete(calender_url, params={**calender_param, 'id': entry['id']})
                    print("del_response:", del_response)
                    if del_response.status_code == 200:
                        details = format_entry_details(entry)
                        response = f'{details} has been deleted.'
                    else:
                        response = 'Failed to delete the event'
        elif llm_response.request == 'put':
            if ids:
                event_data = llm_response.calender.model_dump()
                print("\nevent data:", event_data)
                put_response = requests.put(calender_url, params={**calender_param, 'id': ids[0]}, json=event_data)
                print("put_response:", put_response)
                if put_response.status_code == 200:
                    details = format_entry_details(event_data)
                    response = f'{details} has been updated.'
                    created_entry_id = ids[0]
                else:
                    response = 'Failed to process calender request, please try again.'

    else:
        response = 'Failed to process calender request, please try again.'
    print("calender response:", response)
    return response, {'calendar_entry': created_entry_id}


def voice_assistant():
    chat_history = st.session_state.chat_history

    if chat_history.chats:
        st.subheader("Chat history")
        for chat in chat_history.chats[-4:]:
            st.text(f"👱  : {chat.usr_request}")
            st.text(f"🤖  : {chat.chat_response}")
            st.divider()

    audio_file = st.file_uploader("Upload audio", type=["mp3"], key=f"uploader_{st.session_state.uploader_key}")

    if st.button("Record Audio"):
        start_time = time.time()

        if audio_file:
            # use file for audio
            audio_np, _ = librosa.load(
                audio_file,
                sr=16_000,
                mono=True
            )

            audio = audio_np.astype(np.float32)
        else:
            # record audio
            audio = record_audio()
        with st.spinner("Transcribing!!"):
            text = predict_audio(audio)

        print(text)
        st.text(text)

        last_city = chat_history.get_last_city()
        last_calender_id = chat_history.get_last_appointment()
        # LLM testing
        result = extractor_llm_model.chat(text, last_city, last_calender_id)
        print("LLm response:", result)

        if result.intent == 'calender':
            response, entries = calender_process(text, result, chat_history)
        elif result.intent == 'weather':
            response, entries = weather_process(text, result)

        chat_history.add(
            usr_request=text,
            chat_response=response,
            intent=result.intent,
            entries=entries,
        )

        # tts the response
        audio = tts_model.speak(response)
        print('Total tts infer time:', time.time() - start_time)

        # rerun record
        st.session_state.uploader_key += 1
        st.rerun()

    if st.button("Clear History"):
        chat_history.clear()
        st.session_state.uploader_key += 1
        st.rerun()


# try:
#     voice_assistant()
# except Exception as e:
#     print(e)
#     st.rerun()

voice_assistant()
