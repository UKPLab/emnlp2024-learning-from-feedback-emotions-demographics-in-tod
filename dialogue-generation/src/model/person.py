from dataclasses import dataclass
from typing import Dict

@dataclass
class Person:
    name: str = ''
    age: str = ''
    language: str = ''
    background: str = ''

    @classmethod
    def from_dict(cls, person:Dict):
        name = person['name'] if 'name' in person else ''
        age = person['age'] if 'age' in person else ''
        language = person['language'] if 'language' in person else ''
        background = person['background'] if 'background' in person else ''
        return cls(name, age, language, background)