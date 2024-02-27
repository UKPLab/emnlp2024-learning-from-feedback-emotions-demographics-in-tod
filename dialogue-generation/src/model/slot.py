from dataclasses import dataclass

@dataclass
class Slot:
    span_type: str
    span: str
    span_character_start_position: int
    