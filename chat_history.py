from dataclasses import dataclass
from datetime import datetime

@dataclass
class ChatData: 
    '''chat stores: user request, chat response, intent type '''

    time: str
    usr_request: str
    chat_response: str
    intent: str
    entries: dict

class ChatHistory:

    def __init__(self):
        self.chats = []

        self.last_weather_city = None
        self.last_calender_entry = None
        self.last_intent = None

    def add(self, usr_request, chat_response, intent=None, entries=None):

        data = ChatData(
            time=datetime.now(),
            usr_request=usr_request,
            chat_response=chat_response,
            intent=intent,
            entries=entries
        )
        self.chats.append(data)
        self.last_intent = intent

        if entries:
            if 'city' in entries:
                self.last_weather_city = entries['city']
            if 'calendar_entry' in entries:
                self.last_calender_entry = entries['calendar_entry']


    def clear(self):
        self.chats = []

    def get_last_contents(self):
        return{self.chats}
    
    def get_last_city(self):
        return self.last_weather_city
    
    def get_last_appointment(self):
        return self.last_calender_entry