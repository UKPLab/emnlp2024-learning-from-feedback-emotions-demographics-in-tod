'''
    openai_handler.py -- Script for classes that manage access to OpenAI's api.
'''

import openai
import time
import json
import re
from model.open_ai_response import OpenAIResponse
from model.prompts import PromptResult
from openai.error import RateLimitError, APIError, APIConnectionError 
from openai.openai_object import OpenAIObject
from utils.errors import CallOpenAIException, InvalidOpenAIResponse
from typing import Tuple

class OpenAi:
    def __init__(self,
                 api_key_file:str,
                 retries:int=10,
                 model:str="gpt-3.5-turbo") -> None:
        with open(api_key_file, 'r', encoding='utf-8') as file:
            openai.api_key = re.sub('\n', '', file.readline())
        self.model = model
        self.retries = retries

    def __call_openai(self, 
                    prompt:str,
                    retries:int=0) -> Tuple[OpenAIObject, int]:
        if retries < 1:
            raise CallOpenAIException(retries=self.retries)
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            if isinstance(response, OpenAIObject):
                return response, retries
            raise InvalidOpenAIResponse(response)
        except (RateLimitError, APIError, APIConnectionError):
            # if an API error occurs, it's usually a BadGateway error (which is 
            # a recurring problem for OpenAI, they're working on it)
            time.sleep(30)
            return self.__call_openai(prompt, retries-1)
        
    def prompt_chatgpt(self, prompt: str, result_type:str, retries=None)\
        -> PromptResult:
        _retries = retries if retries else self.retries
        _response, retries = self.__call_openai(prompt, _retries)
        response = dict()
        try:
            response = OpenAIResponse.from_json(_response)\
                if result_type == 'json'\
                else OpenAIResponse.from_dialog(_response)
        except json.decoder.JSONDecodeError as error:
            print('There was an exception while decoding response: ')
            print(error)
            print(_response['choices'][0]['message']['content'])
            print('Will try again...')
            return self.prompt_chatgpt(prompt, result_type, _retries-1)
        return PromptResult(prompt, response)
