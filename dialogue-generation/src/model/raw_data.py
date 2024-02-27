from dataclasses import dataclass, field
from model.prompts import PromptResults
from typing import Dict

@dataclass
class RawData:
    raw_dialog: dict = field(default_factory=dict)
    documents: dict = field(default_factory=dict)
    persona: dict = field(default_factory=dict)
    num_user_uts: int = 0
    prompts: PromptResults = field(default_factory=PromptResults)
    stringified_dialog: str = ''

    @staticmethod
    def from_json(raw_data:Dict) -> 'RawData':
        return RawData(raw_data['raw_dialog'],
            raw_data['documents'],
            raw_data['persona'],
            raw_data['num_user_uts'])        

    def __post_init__(self) -> None:
        dialog = []
        for ut in self.raw_dialog:            
            prefix = '[Robot]:' if 'robot' in\
                ut['subject'].lower() else '[Person]:'
            _ut = prefix + ' ' + ut['text']
            dialog.append(_ut)
        self.stringified_dialog = '\n '.join(dialog)
