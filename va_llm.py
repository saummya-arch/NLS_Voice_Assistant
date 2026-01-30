from ollama import chat
from typing import Literal

from pydantic import BaseModel


class Calender(BaseModel):
    request_type: Literal['get', 'post', 'put', 'delete']
    create_new_entry: bool
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
    def chat(self, text: str, last_city: str, last_calendar_id: str):

        last_respone = 'No previous context'
        if last_city:
            last_respone = f"The last mentioned city was {last_city}. If the user says there, use this city."
        if last_calendar_id:
            last_respone = f"The last created or modified calender entry was: {last_calendar_id}. If user says 'previous' or 'previously created', use this ID."
        
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
                
                For CALENDAR intent, determine request_type:
                - "post" = add/create/schedule new appointment
                - "get" = retrieve/show/find existing appointment
                - "put" = update/change/modify existing appointment
                - "delete" = remove/cancel existing appointment

                Content from previous conversation:
                {last_respone}

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
    request: Literal['get', 'put', 'delete', 'post']
    ids: list[int]
    calender: Calender


class ModifyCalenderLLM:
    def chat(self, request: str, data: dict, last_calendar_id: int = None):

        last_respone = ''
        if last_calendar_id:
            last_respone = f"The previously created or modified calender entry ID was: {last_calendar_id}. If request mentions says 'previous' or 'previously created', use this ID."
        
        response = chat(messages=[
            {
                'role': 'user',
                'content': f'''
                request type is either get, put or delete API call.
                Output the id(s) of the data that best fits the answer.
                Id with the highest value is the last created event.

                {last_respone}

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
    def chat(self, question: str, keyword, is_calender_event:bool = False):
        response = chat(
            messages=[
                {
                    'role': 'user',
                    'content': f'''
                     Generate an appropriate clean reply to the given question. 
                     Use the Answer keywords to generate the answer.
                     Answer should be plain text.
                     
                     For CALENDAR events:
                     - Summarize calendar events in one short plain-text paragraph.
                     - Prioritize dates and main activities.
                     - It is fine if some events or details are omitted.
                     - When days are referenced please check the date information.
                     
                     Please keep the text clean and DO NOT highlight the keywords in the answer.
                     Text Length max 30 words.
                     
                     Question
                     {question}
                    
                     Answer Keywords
                     {keyword}
                     
                     Is the question for a calendar event: {is_calender_event}
                     '''
                }
            ],
            model='ministral-3',
            format=Reply.model_json_schema())
        return Reply.model_validate_json(response.message.content)
