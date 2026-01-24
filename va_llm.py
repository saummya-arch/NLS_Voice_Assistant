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
    # day: Literal['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'] | None
    day: str | None


class Response(BaseModel):
    intent: Literal['weather', 'calender', None]
    calender: Calender
    weather: Weather


class ExtractorLLM:
    def chat(self, text: str):
        response = chat(messages=[
            {
                'role': 'user',
                'content': f'''
                Determine whether request is calender or weather intent.
                Extract the relevant calender or weather information from the given text.
                Dates and Time use yyyy-mm-ddThh:mm.
                If only Dates are given use yyyy-mm-dd.
                Provide no output for missing information.
                For weather location use only proper city name, no arbitrary location.
                Do not pick dates that have passed when weekday is given, pick next date available date.
                Text:
                {text}
                '''
            }
        ],
            model='ministral-3',
            format=Response.model_json_schema())
        result = Response.model_validate_json(response.message.content)
        return result


class Reply(BaseModel):
    answer: str | None


class ReplyLLM:
    def chat(self, question: str, keyword):
        response = chat(
            messages=[
                {
                    'role': 'user',
                    'content': f'''
                     Generate an appropriate clean reply to the given question. 
                     Use the Answer keywords to generate the answer.
                     Please keep the text clean and DO NOT highlight the keywords in the answer.
                     
                     Question
                     {question}
                    
                     Answer Keywords
                     {keyword}
                     '''
                }
            ],
            model='ministral-3',
            format=Reply.model_json_schema())
        return Reply.model_validate_json(response.message.content)
