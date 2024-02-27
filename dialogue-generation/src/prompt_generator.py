'''
    prompt_generator.py -- Script file for prompt_config and prompt_generator
    classes.
'''

import yaml
import json
import re
from typing import List, Dict, Tuple
from pathlib import Path
import random
from model.dialog_container import DialogContainer

class PromptConfig:
    """Class for loading config and json files.
    """

    def __init__(self, config_path:Path) -> None:
        with open(config_path, 'r', encoding='utf-8') as y:
            try: 
                self.config = yaml.safe_load(y)
            except yaml.YAMLError as error:
                print(f"Exception while loading yaml file: {error}.")
                exit()

        self._dialog =\
            self._read_json(Path(self.config['dialogs']))
        self._error_dialogs =\
            self._read_json(Path(self.config['error_dialogs']))        
        self._annotations =\
            {task:self._read_json(Path(file)) for task, file in 
             self.config['annotations'].items()}
        self._documents =\
            {task:self._read_documents(Path(file)) for task, file in 
             self.config['documents'].items()}        
        self._names = self._read_json(Path(self.config['names']))
        self._occupations = self._read_plain_text(Path(self.config['occupations']))

    @property
    def dialog(self) -> Dict:
        return self._dialog
    
    @property
    def error_dialogs(self) -> Dict:
        return self._error_dialogs
        
    @property
    def annotations(self) -> Dict:
        return self._annotations
    
    @property
    def documents(self) -> Dict:
        return self._documents

    @property
    def names(self) -> Dict:
        return self._names
    
    @property
    def occupations(self) -> List:
        return self._occupations
    
    def _read_plain_text(self, file_path:Path) -> List:
        with open(file_path, 'r', encoding='utf-8') as fi:
            return [row.strip().lower() for row in fi]
    
    def _read_json(self, file_path:Path) -> Dict:
        with open(file_path, 'r', encoding='utf-8') as file:            
            return json.load(file)
        
    def _read_documents(self, file_path:Path) -> List:
        """Returns the content of all files in file_path and 
           returns their content as list.

        Args:
            file_path (Path): file_path

        Returns:
            List: read content
        """
        documents = []
        for file in file_path.glob('**/*'):
            with open(file, 'r', encoding='utf-8') as f:
                documents += [' '.join(f.readlines()).strip().replace('\n', '')]
        return documents

class PromptGenerator:
    """Class for generating prompts
    """

    def __init__ (self,
                  config_path: Path) -> None:  
        self.config = PromptConfig(config_path)

    def fix_dialog_prompt(self, task_names:List[str], 
        subject:str, dialog:str, add_info:List[str]) -> str:

        config = self.config.error_dialogs
        _task_names = ', '.join(task_names) if len(task_names) > 1\
            else task_names[0]
        
        instruction = config['instruction_fix_robot']\
            if subject.lower() == 'robot' else config['instruction_fix_user']
        instruction = re.sub(r'{{task_name}}', _task_names, instruction)
        instruction = re.sub(r'{{dialog}}', dialog, instruction)

        if subject.lower() == 'robot':
            instruction = re.sub(r'{{document}}', add_info[0], instruction)
        else:
            phrases = list(config['user_reaction_types'].values())
            # leave out the first one, since it is rather unnatural in this 
            # case...
            phrases = ', '.join(phrases[0]['examples'][1:])
            instruction = re.sub(r'{{sub_robot_response}}', add_info[1], 
                instruction)
            instruction = re.sub(r'{{prev_robot_response}}', add_info[0], 
                instruction)     
            instruction = re.sub(r'{{sub_person_utterance}}', add_info[2], 
                instruction)
            instruction = re.sub(r'{{phrases}}', phrases, instruction)
            
        
        return instruction

    def error_scenario_prompt(self, num_errors:int, 
        task_names:List[str], topic:str) -> Tuple[str, Dict]:
        """Create a prompt for generating error scenarios.

        Args:
            num_errors (str): Number of errors to generate scenarios for.

        Returns:
            str: Prompt for generation of error scenarios.
        """

        config = self.config.error_dialogs
        instruction = config['instruction_scenario']

        # randomly sample error and user reaction types
        errors = random.sample(range(0, 9), num_errors)
        user_reactions = random.sample(range(0, 5), num_errors)        
        
        # get items to replace placeholders in the instrcution
        errors = [e for i, e in enumerate(config['errors'].items()) 
            if i in errors]
        user_reactions = [ur for i, ur in 
            enumerate(config['user_reaction_types'].items()) 
            if i in user_reactions]
        
        # generate error descriptions with examples
        error_descs = []
        for error in errors:
            desc = error[1]['desc']            
            idx = random.randint(0, len(error[1]['examples'])-1)            
            ex = error[1]['examples'][idx]
            error_descs += [f'{desc} For example: {ex}']
        error_descs = '\n\n'.join(error_descs)

        # generate user response types with examples
        ur_descs = []    
        for ur in user_reactions:
            desc = ur[1]['desc']
            idx = random.randint(0, len(ur[1]['examples'])-1)
            ex = ur[1]['examples'][idx]
            ur_descs += [f'{desc}, e.g., {ex}']
        ur_descs = ', '.join(ur_descs)

        error_names = ', '.join(e[0] for e in errors)    

        # get task descriptions
        tasks =\
            '\n\n'.join([self.config.dialog['tasks']['question_answering_error_scenario' if task == 'question_answering' else task]
            for task in task_names])
        
        # build instruction
        instruction = re.sub(r'{{error_type_names}}', error_names, instruction)
        instruction = re.sub(r'{{list_of_error_types}}', error_descs, 
            instruction)
        instruction = re.sub(r'{{num_scenarios}}', str(num_errors), instruction)
        instruction = re.sub(r'{{user_reaction_types}}', ur_descs, instruction)
        instruction = re.sub(r'{{tasks}}', tasks, instruction)
        instruction, _documents = self._add_documents(instruction, task_names,
            topic)
        
        return instruction, _documents
    
    def error_dialog_prompt(self, error_scenarios:Dict, task_names:List[str],
        documents:Dict) -> Tuple[str, dict, dict]:
        
        # Do all the preparing stuff, i.e., generate persona, language, tasks 
        # prompt
        persona, _persona, _name, language, starting_actor, tasks =\
            self._prepare_dialog_prompt(task_names)
        
        # generate scenarios prompt
        scenario_prompt = '\n\n'.join(sc[0] + ': ' + sc[1]['scenario'] 
            for i, sc in enumerate(error_scenarios.items()))
        
        # build instruction
        instruction = self.config.error_dialogs['instruction_dialog']
        instruction = re.sub(r'{{num_scenarios}}', 
            str(len(list(error_scenarios.keys()))), instruction)
        instruction = re.sub(r'{{error_scenarios}}', scenario_prompt, 
            instruction)
        #instruction = re.sub(r'{{tasks}}', ','.join(task_names), instruction)
        instruction =\
            re.sub(r'{{starting_actor}}', starting_actor, instruction)
        instruction = re.sub(r'{{task_descriptions}}', tasks, instruction)
        instruction = re.sub(r'{{persona}}', persona, instruction)
        instruction = re.sub(r'{{language}}', language, instruction)
        instruction = re.sub(r'{{name}}', _name, instruction)

        if documents:
            _documents = json.dumps(documents)
            instruction = re.sub(r'{{documents}}', _documents, instruction)
            instruction = re.sub(r'{{doc}}', list(documents.keys())[0],
                instruction)
        
        return instruction, _persona, documents        
            
    def dialog_prompt(self, task_names:List[str], topic:str)\
        -> Tuple[str, dict, dict]:
        """Create a prompt for generating a dialog for the tasks from task_names

        Args:
            task_names (List[str]): list of tasks you want to combine

        Returns:
            Tuple[str, str]: instruction for dialog generation, persona
        """

        # Do all the preparing stuff, i.e., generate persona, language, tasks 
        # prompt
        persona, _persona, _name, language, starting_actor, tasks =\
            self._prepare_dialog_prompt(task_names)
        
        # build instruction
        instruction = self.config.dialog['instruction']
        instruction = re.sub(r'{{tasks}}', ','.join(task_names), instruction)
        instruction =\
            re.sub(r'{{starting_actor}}', starting_actor, instruction)
        instruction = re.sub(r'{{task_descriptions}}', tasks, instruction)
        instruction = re.sub(r'{{persona}}', persona, instruction)
        instruction = re.sub(r'{{language}}', language, instruction)
        instruction = re.sub(r'{{name}}', _name, instruction)
        instruction, _documents =\
            self._add_documents(instruction, task_names, topic)
        return instruction, _persona, _documents   
    
    def slot_prompt(self, task:str, dialog:str) -> str:
        instruction = self.config.annotations['slots'][task]
        instruction = re.sub(r'{{dialog}}', dialog, instruction)
        return instruction
    
    def other_annotation_prompt(self, type:str, dialog:DialogContainer) -> str:
        """This is for creating a prompt for all other annotation types, e.g., 
            emotions
        

        Args:
            type (str): annotation type
            dialog (DialogContainer): the dialog you want to annotate

        Returns:
            str: prompt
        """

        num_uts = str(len(dialog.raw_data.raw_dialog))
        num_uts_role = str(dialog.raw_data.num_user_uts)
        instruction = self.config.annotations['additional'][type]
        instruction = re.sub(r'{{num_uts}}', num_uts, instruction)
        instruction = re.sub(r'{{num_uts_role}}', num_uts_role, instruction)
        instruction =\
            re.sub(r'{{dialog}}', dialog.raw_data.stringified_dialog,
                instruction)        
        return instruction

    def _random(self, attributes:List[str]) -> str:        
        return attributes[random.randint(0, len(attributes)-1)]
    
    def _prepare_dialog_prompt(self, task_names:List[str])\
        -> Tuple[str, Dict, str, str, str, str]:
        # randomly choose language        
        language = self._random(self.config.dialog['language'])
                
        # randomly choose persona
        _age = self._random(self.config.dialog['personas']['ages'])
        _job = 'pupil' if _age == '6 and 15'\
            else self._random(self.config.occupations)
            # else self._random(self.config.dialog['personas']['jobs'])
        _gender = self._random(self.config.dialog['personas']['gender'])
        _name = self._random(self.config.names['names']['male']) if _gender == 'male' else self._random(self.config.names['names']['female'])
        _persona = {'gender': _gender, 'job': _job, 'age': _age, 'name': _name,
            'language': language}
        persona = ' '.join([_gender, _job, 'between', _age, 'years old'])
        
        # randomly choose starting actor
        starting_actor = self._random(self.config.dialog['starting_actors'])
        
        # get the task descriptions
        task_names = ['greeting'] + task_names
        tasks =\
            '\n\n'.join([self.config.dialog['tasks'][task]
                for task in task_names])        
        return persona, _persona, _name, language, starting_actor, tasks

    def _add_documents(self, instruction:str, task_names:List[str],
        topic:str) -> Tuple[str, Dict]:        
        _documents = dict()
        if 'question_answering' in task_names:
            _documents = self.config.documents[topic]
            _documents = random.sample(_documents, 2)            
            _documents = {'document_' + str(k):v for k, v 
                in enumerate(_documents)}
            documents = json.dumps(_documents)
            instruction = re.sub(r'{{documents}}', documents, instruction)
            instruction = re.sub(r'{{doc}}', list(_documents.keys())[0],
                instruction)
        return instruction, _documents

if __name__ == "__main__":
    prompt_generator = PromptGenerator(Path('prompt_config.yml'))
    #print(prompt_generator.dialog_prompt(['recharge_a_prepaid_phone', 'request_ticket']))
    #print(prompt_generator.intent_prompt(['parcel_choice'], 'test'))