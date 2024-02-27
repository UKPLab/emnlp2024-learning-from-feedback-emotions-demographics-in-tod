from __future__ import annotations
import re
from datetime import datetime
from dataclasses import dataclass, field
from model.slot import Slot
from model.feedback import Feedback
from typing import List, Dict

@dataclass
class Record:
    attachments: list[str] = field(default_factory = list)
    intent: str = ''
    slots: list[Slot] = field(default_factory = lambda: [])
    subject: str = ''
    text: str = ''
    timestamp: str = datetime.now().isoformat()
    feedbacks: list[Feedback] = field(default_factory = list)        
    emotion: str = ''
    error_type: str = ''
    error_scenario: str = ''
    user_reaction_type: str = ''

    def stringify(self) -> str:
        prefix = '[Person]: ' if self.subject != 'agent' else '[Robot]: '
        return ' '.join([prefix, self.text])

    @staticmethod
    def from_dict(record: Dict) -> Record:
        slots = [
            Slot(slot['span_type'], slot['span'], 
                 slot['span_character_start_position']) 
            for slot in record['slots']]
        # feedback = ... not used yet
        return Record(record['attachments'],
            record['intent'],
            slots,
            record['subject'],
            record['text'],
            record['timestamp'],
            [],                        
            record['emotion'],
            record['error_type'],
            record['error_scenario'],
            record['user_reaction_type'])
        
    @staticmethod
    def create(utt: dict, **kwargs) -> Record:
        # subject = 'agent' if 'service robot' in utt['subject'].lower()\
        subject = 'agent' if 'robot' in utt['subject'].lower()\
            else 'user'
        new_record = Record(subject=subject, text=utt['text'])
        if 'task' in utt:
            new_record.intent=utt['task']
        elif 'intent' in utt:
            new_record.intent=utt['intent']
        if 'attachments' in utt and new_record.intent == 'question_answering':
            new_record.attachments = utt['attachments']
        if kwargs:
            if subject == 'agent':
                    new_record.error_type = kwargs['error_type']
                    new_record.error_scenario = kwargs['error_description']
            else:
                new_record.user_reaction_type = kwargs['user_reaction']
        return new_record
        
    def assign_slots(self, slots: Dict[str, str]|list) -> None:
        if isinstance(slots, list):
            for slot in slots:
                self._assign_slots(slot)
        else:
            self._assign_slots(slots)
    
    def _assign_slots(self, slots: Dict[str, str]) -> None:
        
        for key, value in slots.items():
            _value = re.sub(r'[^a-zA-Z0-9]', ' ', str(value).lower())
            _value = re.sub(r' +', ' ', _value)

            _text = re.sub(r'[^a-zA-Z0-9]', ' ', self.text.lower())
            _text = re.sub(r' +', ' ', _text)           

            if len(re.findall(f'(?:{_value})+', _text)) > 0:                
                # check that it is not set twice for the same utt
                if len(value) > 0 and value not in\
                    [slot.span for slot in self.slots]:
                    self.slots.append(Slot(key, str(value), _text.index(_value)))