from dataclasses import dataclass, field

@dataclass
class Feedback:
    feedback: str
    code: str
    source: int
    correction: str
    wrong: str