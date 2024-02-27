import json
from typing import Dict
from dataclasses import dataclass, asdict, field
from model.session import Session
import os

@dataclass
class Dialog:
   session: Session = field(default_factory=Session)
   group: str = 'dialogue'

   @staticmethod
   def from_json(dialog:Dict) -> 'Dialog':
      session = Session.from_json(dialog['session'])
      return Dialog(session=session, 
         group=dialog['group'])
   
   def stringify(self) -> str:
      return '\n'.join([r.stringify() for r in self.session.records])

   def dump(self, filename:str) -> None:
      with open(filename, 'w', encoding='utf-8') as file:
         json.dump(asdict(self), file)

   def delete(self, filename:str) -> None:
      os.remove(filename)
   