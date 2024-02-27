'''
    errors.py -- Script for application specific error classes.
'''

from typing import List, Any

class CallOpenAIException(Exception):
    """Exception if the max number of retries exceeded while calling OpenAI for 
       any request.

    Args:
        Exception (_type_): Abstract Exception class.
    """

    def __init__(self,
                message:str|None=None,
                errors:List[Exception]|None=None,
                retries:int=10) -> None:
        if message is None:
            self.message = f"Failed {retries} times to generate a dialogue."\
                + " Something is wrong!"
        if errors is not None:
            self.errors = errors

class InvalidOpenAIResponse(Exception):

    def __init__(self,
                response: Any) -> None:    
        self.message = "Received invalid response object: "\
            + f"{type(response)}. \n String representation: {str(response)}"
        
class MissingArgumentException(Exception):
        
        def __init__(self,
                arg: str) -> None:    
            self.message = f"Argument missing: {arg}."

class MissingScenariosException(Exception):
        
        def __init__(self) -> None:    
            self.message = "Error field has value but no scenarios are"\
                + " available!"
            
class NoErrorsInDialogException(Exception):
        
        def __init__(self, filename: str) -> None:    
            self.message = f"There are no errors in {filename}. Nothing to fix!"