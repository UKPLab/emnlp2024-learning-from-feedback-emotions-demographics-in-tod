'''
    dialog_container.py -- Script contains container class for dialog objects (dialogs and everything related)
'''

import os
import json

from typing import Optional, Dict
from dataclasses import dataclass, field, asdict
from model.dialog import Dialog
from model.record import Record
from model.raw_data import RawData
from model.session import Session
from model.prompts import PromptResults
from model.open_ai_response import OpenAIResponse
from utils.errors import MissingScenariosException

@dataclass
class DialogContainer:
    dialog: Dialog = field(default_factory=Dialog)
    raw_data: RawData = field(default_factory=RawData)
    filename: str = ''
    
    def __init__(self,
                 filename: str,                 
                 response: Optional[OpenAIResponse],
                 **kwargs):
        self.filename = str(filename)
        if response:            
            if len(response.dialog.dialog) == 0:
                raise Exception('Raw dialog length empty!')  
            raw_dialog = response.dialog.dialog
            self.raw_data = RawData(raw_dialog=raw_dialog)
            self.dialog = Dialog()                
            self.kwargs = kwargs
            self.__post_init__()
    
    @staticmethod
    def from_json(filename:str) -> 'DialogContainer':
        if not filename.endswith('json'):
            # raise exception
            pass
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            container = DialogContainer(filename, None)
            container.dialog = Dialog.from_json(data['dialog'])
            container.raw_data = RawData.from_json(data['raw_data'])            
            return container

    def __post_init__(self) -> None:
        for utterance in self.raw_data.raw_dialog:
            if 'error_scenario' in utterance\
                and len(utterance['error_scenario']) > 0:
                if not 'scenarios' in self.kwargs:
                    raise MissingScenariosException()
                scenario = self.kwargs['scenarios'][utterance['error_scenario']]
                error_type = scenario['error_type']
                error_description = scenario['scenario']
                user_reaction = scenario['user_reaction_type']
                record = Record.create(utterance, error_type=error_type,
                    error_description=error_description,
                    user_reaction=user_reaction)
            else:
                record = Record.create(utterance)
            self.dialog.session.records.append(record)        
        self.raw_data.num_user_uts =\
            len([record for record in self.dialog.session.records 
                 if record.subject.lower() == 'user'])        
        self.dump()

    @property
    def session(self) -> Session:
        return self.dialog.session
    
    @property
    def prompts(self) -> PromptResults:
        return self.raw_data.prompts
    
    @property
    def is_valid(self) -> bool:
        return len(self.raw_data.raw_dialog) > 1

    def dump(self) -> None:
        with open(self.filename, 'w', encoding='utf-8') as file:
            json.dump(asdict(self), file)

    def delete(self) -> None:
        try:
            os.remove(self.filename)    
        except FileNotFoundError:
            pass
