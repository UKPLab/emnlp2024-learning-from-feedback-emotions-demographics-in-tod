import json
import os

from torch.utils.data import Dataset
from typing import Dict, List
from transformers import AutoTokenizer


class PosteDataset(Dataset):
    def __init__(self, path:str, tokenizer:AutoTokenizer, model:str, 
        _type:str, args):
        self.max_input_token = args.max_input_token
        self.max_target_token = args.max_target_token
        self.tokenizer = tokenizer
        self.model = model
        self.type = _type
        self.user_personas = args.user_personas
        self.additional_annotations = args.additional_annotations
        self.error_text = args.error_text
        self.user_reaction = args.user_reaction
        self.emotion = args.emotion
        self.actions = args.actions

        if os.path.isdir(path):
            self.samples = []
            for subdir, _, files in os.walk(path):
                for file in files:
                    f_path = os.path.join(subdir, file)
                    with open(f_path, 'r', encoding='utf-8')  as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.samples += data
                        else:
                            self.samples += [data]

        else:
            raise Exception('Dir expected!')   
        
    def __get_template_gpt2(self) -> str:    
        if self.user_personas and self.additional_annotations:
            template =  "<knowledge> [knowledge] <user_persona> [persona] "\
                + "<user_emotion> [emotion] <user_gesture> [gesture] "\
                + "<system_action> [action] <dialog> [history] "\
                + "<intent> [intent] <slots> [slots] "\
                + "<system> [target] <|endoftext|>"
        if self.user_personas and self.emotion:
            template =  "<knowledge> [knowledge] <user_persona> [persona] "\
                + "<user_emotion> [emotion] <dialog> [history] "\
                + "<intent> [intent] <slots> [slots] "\
                + "<system> [target] <|endoftext|>"            
        elif self.user_personas:
            template =  "<knowledge> [knowledge] <user_persona> [persona] "\
                + "<dialog> [history] <intent> [intent] <slots> [slots] "\
                + "<system> [target] <|endoftext|>"
        elif self.additional_annotations:
            template =  "<knowledge> [knowledge] <user_emotion> [emotion] "\
                + "<user_gesture> [gesture] <system_action> [action]"\
                + "<dialog> [history] <intent> [intent] <slots> [slots] "\
                + "<system> [target] <|endoftext|>"        
        elif self.actions:
            template =  "<knowledge> [knowledge] "\
                + "<user_gesture> [gesture] <system_action> [action]"\
                + "<dialog> [history] <intent> [intent] <slots> [slots] "\
                + "<system> [target] <|endoftext|>"
        elif self.emotion:
            template =  "<knowledge> [knowledge] <user_emotion> [emotion] "\
                + "<dialog> [history] <intent> [intent] <slots> [slots] "\
                + "<system> [target] <|endoftext|>"        
        else:
            template = "<knowledge> [knowledge] <dialog> [history] "\
                + "<intent> [intent] <slots> [slots] "\
                + "<system> [target] <|endoftext|>"

        if self.user_reaction and self.error_text:
            template = template.replace('<dialog> [history]', '<error_text> [error_text] <user_reaction> [user_reaction] <dialog> [history]')
        elif self.user_reaction:
            template = template.replace('<dialog> [history]', '<user_reaction> [user_reaction] <dialog> [history]')
        elif self.error_text:
            template = template.replace('<dialog> [history]', '<error_text> [error_text] <dialog> [history]')

        return template
        
    def __get_template_flan_t5(self) -> str:    
        if self.user_personas and self.additional_annotations:
            template =  "<knowledge> [knowledge] <user_persona> [persona] "\
                + "<user_emotion> [emotion] <user_gesture> [gesture] "\
                + "<system_action> [action] <dialog> [history] "\
                + "<intent> [intent] <slots> [slots] "\
                + "<system> [target]"
        elif self.user_personas and self.emotion:
            template =  "<knowledge> [knowledge] <user_persona> [persona] "\
                + "<user_emotion> [emotion] <dialog> [history] "\
                + "<intent> [intent] <slots> [slots] "\
                + "<system> [target]"                    
        elif self.user_personas:
            template = "<knowledge> [knowledge] <user_persona> [persona] "\
                + "<dialog> [history] <intent> [intent] <slots> [slots] "\
                + "<system> [target]"
        elif self.additional_annotations:
            template = "<knowledge> [knowledge] <user_emotion> [emotion] "\
                + "<user_gesture> [gesture] <system_action> [action]"\
                + "<dialog> [history] <intent> [intent] <slots> [slots] "\
                + "<system> [target]"
        elif self.actions:
            template = "<knowledge> [knowledge] "\
                + "<user_gesture> [gesture] <system_action> [action]"\
                + "<dialog> [history] <intent> [intent] <slots> [slots] "\
                + "<system> [target]"            
        elif self.emotion:
            template = "<knowledge> [knowledge] <user_emotion> [emotion] "\
                + "<dialog> [history] <intent> [intent] <slots> [slots] "\
                + "<system> [target]"             
        else:
            template = "<knowledge> [knowledge] <dialog> [history] "\
                + "<intent> [intent] <slots> [slots] "\
                + "<system> [target]"

        if self.user_reaction and self.error_text:
            template = template.replace('<dialog> [history]', '<error_text> [error_text] <user_reaction> [user_reaction] <dialog> [history]')
        elif self.user_reaction:
            template = template.replace('<dialog> [history]', '<user_reaction> [user_reaction] <dialog> [history]')
        elif self.error_text:
            template = template.replace('<dialog> [history]', '<error_text> [error_text] <dialog> [history]')

        return template                

    def __shorten(self, text: str, max_length: int, offset: str = 'left'):
        tokenized = self.tokenizer.encode(text)        
        if len(tokenized) > max_length:
            tokenized = tokenized[(max_length-len(tokenized))*(-1):]\
                if offset == 'left' else tokenized[:max_length]
        return self.tokenizer.decode(tokenized)                

    def __generate_sequence(self, sample:dict) -> str:

            sequence = self.__get_template_flan_t5() if 'T5' in self.model\ 
                else self.__get_template_gpt2()

            hist_length = 385
            if self.user_reaction:
                hist_length -= 25
            if self.error_text:
                hist_length -= 25

            history = self.__shorten('<system>'.join(sample['context'].split('<system>')[:-1]), 385)
            intent = self.__shorten(sample['intent'], 10, 'right')
            slots = self.__shorten(sample['slots'], 50, 'right')
            target = self.__shorten(sample['target'], 75, 'right')
        
            if len(sample['documents']) > 0:
                knowledge = ' '.join(sample['documents'])
                _knowledge = self.__shorten(knowledge, 385, 'right')
                sequence = sequence.replace('[knowledge]', _knowledge)
            else:
                sequence = sequence.replace('[knowledge]', '')

            sequence = sequence.replace('[history]', history)
            sequence = sequence.replace('[intent]', intent)
            sequence = sequence.replace('[slots]', slots)
            sequence = sequence.replace('[target]', target)

            if self.user_personas:
                persona = self.__shorten(sample['user_persona'], 14, 'right')
                sequence = sequence.replace('[persona]', persona)

            if self.emotion:
                emotion = self.__shorten(sample['emotion'], 3, 'right')
                sequence = sequence.replace('[emotion]', emotion)

            if self.actions:
                gesture = self.__shorten(sample['gesture'], 15, 'right')
                action = self.__shorten(sample['action'], 15, 'right')
                sequence = sequence.replace('[action]', action)
                sequence = sequence.replace('[gesture]', gesture)
                
            if self.additional_annotations:
                emotion = self.__shorten(sample['emotion'], 3, 'right')
                gesture = self.__shorten(sample['gesture'], 15, 'right')
                action = self.__shorten(sample['action'], 15, 'right')
                sequence = sequence.replace('[emotion]', emotion)
                sequence = sequence.replace('[gesture]', gesture)
                sequence = sequence.replace('[action]', action)

            if self.error_text:
                error_text = self.__shorten(sample['error_text'], 25, 'right')
                sequence = sequence.replace('[error_text]', error_text)
            
            if self.user_reaction:
                user_reaction = self.__shorten(sample['user_reaction'], 25,
                    'right')
                sequence = sequence.replace('[user_reaction]', user_reaction)
            
            if 'T5' in self.model:
                x, y = sequence.split('<intent>')            
                x = x.replace('</s>', '') + '</s>'
                y = '<intent> ' + y
                y = y.replace('</s>', '') + '</s>'

                return x, y
            
            else:
                return sequence      
        
    def encode(seq: str, max_length: int):
        return self.tokenizer.encode(seq,        
            #add_special_tokens=True, 
            padding='max_length', 
            max_length=max_length,            
            truncation=True,        
            return_tensors='pt')

    def __getitem__(self, idx):

        if 'T5' in self.model:
            input_seq, target_seq = self.__generate_sequence(self.samples[idx])

            input_tensors = self.tokenizer.encode(input_seq,        
                add_special_tokens=False, 
                padding='max_length',             
                truncation=True,            
                return_tensors='pt')
            target_tensors = self.tokenizer.encode(target_seq,        
                add_special_tokens=False, 
                padding='max_length',             
                truncation=True,            
                return_tensors='pt') 
        else:
            sequence = self.__generate_sequence(self.samples[idx])

            if self.type == 'eval':            
                input, target = sequence.split('<intent>')                
                target_tensors = encode(target, self.max_target_token)
                input_tensors = encode(input, self.max_input_token-1)
                input_tensors += self.tokenizer.encode('<intent>', 
                    return_tensors='pt')                            
            else:            
                input_tensors = encode(sequence, self.max_input_token 
                    + self.max_target_token)
                target_tensors = input_tensors
        
        return {'input': input_tensors, 'target': target_tensors}

    def __len__(self):
        return len(self.samples)
