from dataclasses import dataclass, field, asdict
from datetime import datetime
from model.record import Record
from typing import List, Callable, Dict

from model.slot import Slot
from model.person import Person


@dataclass
class Session:
    groupId: str = ''
    records: list[Record] = field(default_factory = list)    
    sessionId: str = '-'
    label: str = f'Session {datetime.now().strftime("%d/%m%Y, %H:%M")}'
    authorId: str = 'ChatGPT'
    created_at: str = datetime.now().isoformat()
    modified_at: str = datetime.now().isoformat()

    @staticmethod
    def from_json(session: Dict) -> 'Session':
        records = [Record.from_dict(r) for r in session['records']]
        return Session(session['groupId'],
            records,
            session['sessionId'],
            session['label'],
            session['authorId'],
            session['created_at'],
            session['modified_at'])

    def assign_field_values(
            self, 
            values: List[str], 
            key_name: str,  
            role: str,
            idx_error=None) -> None:        
        if not idx_error:
            _records = [record for record in self.records 
                if record.subject.lower() == role]
            for record, value in zip(_records, values):
                setattr(record, key_name, value)
        else:
            _records = [record for record in self.records 
                if record.subject.lower() == role]            
            for i, (record, value) in enumerate(zip(_records, values)):
                if i == idx_error:                    
                    setattr(record, key_name, value)
    
    def assign_slot_values(
            self, 
            raw_slots,
            idx_error=None):        
        slot_keys = list(raw_slots.keys())
        # map slots to utterances        
        #if not idx_error:
        for record in self.records:
            if len(record.slots) == 0:
                if record.subject.lower() == 'user':                        
                    record.assign_slots(raw_slots[slot_keys[0]])                
                else:                
                    record.assign_slots(raw_slots[slot_keys[1]])