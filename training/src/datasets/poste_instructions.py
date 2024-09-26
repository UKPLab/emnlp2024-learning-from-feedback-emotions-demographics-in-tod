import json
import os
import re
from torch.utils.data import Dataset
from typing import Dict, List
from transformers import AutoTokenizer


class PosteDatasetInstructions(Dataset):


    def __init__(self, file:str, tokenizer:AutoTokenizer, type, args):
        self.max_input_token = args.max_input_token
        self.max_target_token = args.max_target_token
        self.tokenizer = tokenizer
        self.type = type

        with open(file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)


    def __getitem__(self, idx):

        def encode(seq: str, max_length: int):
            return self.tokenizer(seq,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
        
        sequence = list(self.samples[idx].values())[0].strip()
                
        if self.type == 'eval':            
            input, target = sequence.split('###Response')
            input += '###Response\n\nUser Intention:'
            
            encoded_target = encode(target, self.max_target_token)['input_ids']
            encoded_input = encode(input, self.max_input_token-1)['input_ids']
   
        else:            
            encoded_input = encode(sequence, self.max_input_token 
                + self.max_target_token)['input_ids']
            encoded_target = encoded_input
        
        return {'input': encoded_input, 'target': encoded_target}

    def __len__(self):
        return len(self.samples)
