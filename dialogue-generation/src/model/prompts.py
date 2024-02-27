from typing import Optional, Dict
from dataclasses import dataclass, field
from model.open_ai_response import OpenAIResponse

@dataclass
class PromptResult:
    prompt: str = ''
    #open_ai_response: Optional[OpenAIResponse|str] = ''    
    open_ai_response: OpenAIResponse = field(default_factory=OpenAIResponse)
    
    #@property
    #def content(self):
    #    if isinstance(self.open_ai_response, OpenAIResponse):
    #        return self.open_ai_response.content
    #    else: 
    #        return ''
    @property
    def content(self) -> Dict:
        return self.open_ai_response.content
        

@dataclass
class PromptResults:
    dialog_generation: PromptResult = field(default_factory=PromptResult)
    slot_generation: list[PromptResult] = field(default_factory=list)
    intent_generation: PromptResult = field(default_factory=PromptResult)    
    emotions_generation: PromptResult = field(default_factory=PromptResult)    
    error_scenarios: PromptResult = field(default_factory=PromptResult)