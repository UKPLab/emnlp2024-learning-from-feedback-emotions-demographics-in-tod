import json
import re
from dataclasses import dataclass, field
from openai.openai_object import OpenAIObject
from typing import Optional, List, Dict
from model.person import Person

@dataclass
class OpenAIDialog:
    person: Person = field(default_factory=Person)
    dialog: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, content:Dict):
        person = Person.from_dict(content['person'])
        dialog = content['dialog']
        return cls(person, dialog)

@dataclass
class OpenAIResponse:
    #content: Optional[dict] = field(default_factory=dict)
    content: dict = field(default_factory=dict)
    #dialog: Optional[OpenAIDialog|str] = ''
    dialog: OpenAIDialog = field(default_factory=OpenAIDialog)
    finish_reason: str = ''
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dialog(cls, res:OpenAIObject):
        finish_reason = res['choices'][0]['finish_reason']
        prompt_tokens = res['usage']['prompt_tokens']
        completion_tokens = res['usage']['completion_tokens']
        total_tokens = res['usage']['total_tokens']
        dialog = OpenAIDialog.from_dict(
            json.loads(res['choices'][0]['message']['content']))
        return cls(dialog=dialog, finish_reason=finish_reason, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)

    @classmethod
    def from_json(cls, res:OpenAIObject):        
        finish_reason = res['choices'][0]['finish_reason']
        prompt_tokens = res['usage']['prompt_tokens']
        completion_tokens = res['usage']['completion_tokens']
        total_tokens = res['usage']['total_tokens']
        cleaned_content = re.sub(r'\\n|\.', '', 
            res['choices'][0]['message']['content'])
        content = json.loads(cleaned_content)
        return cls(content=content, finish_reason=finish_reason, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens) 