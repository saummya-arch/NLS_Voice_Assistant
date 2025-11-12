from typing import Dict, Any

from kokoro import KPipeline
from fastapi import status
import sounddevice as sd
import torch


class TTS:
    def __init__(self):
        self.pipeline = KPipeline(lang_code='a')
        self.voice = 'af_heart'

    def __create_generator(self, text):
        return self.pipeline(text, voice=self.voice, speed=1, split_pattern=r'\n+')

    def speak(self, text: str) -> Dict:
        try:
            if text is None:
                text = 'Did not catch that. Please try again later.'
            generator = self.__create_generator(text)
            for i, (gs,ps, audio) in enumerate(generator):
                # print(gs, ps, audio)
                if isinstance(audio, torch.Tensor):
                    audio = audio.numpy()
                    sd.play(audio, samplerate=24000, blocking=True)
            return {'status': status.HTTP_200_OK}
        except Exception as e:
            print(e)
            return {'status': status.HTTP_400_BAD_REQUEST}
