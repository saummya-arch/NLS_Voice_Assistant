from ollama import chat
from typing import Literal

from pydantic import BaseModel


class Calender(BaseModel):
    create_data: bool
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
                If calender is the intent, new entries are considered new event and all other events are old.
                Text:
                {text}
                '''
            }
        ],
            model='ministral-3',
            format=Response.model_json_schema())
        result = Response.model_validate_json(response.message.content)
        return result


class ModifyCalender(BaseModel):
    request: Literal['get', 'put', 'delete']
    ids: list[int]
    calender: Calender


class ModifyCalenderLLM:
    def chat(self, request: str, data: dict):
        response = chat(messages=[
            {
                'role': 'user',
                'content': f'''
                request type is either get, put or delete API call.
                Output the id(s) of the data that best fits the answer.
                Id with the highest value is the last created event.
                Request:
                {request}
                
                Data:
                {data}
                '''
            }
        ],
            model='ministral-3',
            format=ModifyCalender.model_json_schema())
        return ModifyCalender.model_validate_json(response.message.content)


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
