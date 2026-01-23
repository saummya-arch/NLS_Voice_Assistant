from ollama import chat
from typing import Literal

from pydantic import BaseModel

class Calender(BaseModel):
    intent: Literal['fetch_data', 'update_data']
    start_time: str | None
    end_time: str | None
    title: str | None
    description: str | None
    location: str | None

class Weather(BaseModel):
    city: str | None


class Response(BaseModel):
    intent: Literal['weather', 'calender', 'None']
    calender: Calender
    weather: Weather

class LLM:
    def chat(self, text: str):
        response = chat(messages=[
            {
                'role': 'user',
                'content': f'''
                Determine whether request is calender or weather intent.
                Extract the relevant calender or weather information from the given text.
                DatesTime use yyyy-mm-ddThh:mm.
                Provide no output for missing information.
                For weather location return only city.
                
                Text:
                {text}
                '''
            }
        ],
        model='ministral-3',
        format=Response.model_json_schema())
        result = Response.model_validate_json(response.message.content)
        return result