import os
import datetime
import torch

from pathlib import Path

class DataLogger():

    def __init__(self, data_log:Path, model:str, tokenizer):
        self.tokenizer = tokenizer
        self.model = model

        if not os.path.exists(data_log):
            os.makedirs(data_log)
        
        dt = datetime.datetime.now()
        self.train_log =\
            Path(data_log, dt.strftime('%Y%m%d%H%M%S') + '_train.log')
        self.eval_log =\
            Path(data_log, dt.strftime('%Y%m%d%H%M%S') + '_eval.log')
        self.test_log =\
            Path(data_log, dt.strftime('%Y%m%d%H%M%S') + '_test.log')
        
    def log_train(self, batch:torch.Tensor):
        input_data = self._convert(batch['input'])
        target_data = self._convert(batch['target'])
        with open(self.train_log, 'a+', encoding='utf-8') as f:
            for input, target in zip(input_data, target_data):                
                f.write('INPUT:\n' + input + '\n'*2 + 'TARGET:\n'
                    + target + '\n'*2)
                
    def log_eval_test(self, batch, predictions, mode):
        input_data = self._convert(batch['input'])
        target_data = self._convert(batch['target'])
        predictions = self._convert(predictions, 
            True if 'T5' in self.model else False)
        file = self.eval_log if mode == 'evaluation' else self.test_log
        with open(file, 'a+', encoding='utf-8') as f:
            for inp, pred, tar in zip(input_data, predictions, target_data):
                f.write('INPUT:\n' + inp + '\n'*2 + 'TARGET:\n'
                    + tar + '\n'*2 + 'PREDICTION:\n' + pred
                    + '\n'*2)
                
    def _convert(self, batch:torch.Tensor, skip_special_tokens=False):
        _data = self.tokenizer.batch_decode(batch.squeeze(),
            skip_special_tokens=skip_special_tokens)
        
        data = []
        for dat in _data:
            if 'T5' in self.model:
                dat = dat.split('<pad>')[0]
                dat = dat.split('</s>')[0] + '</s>'                
            else:
                dat = dat.split('<pad>')[0]            
                dat = dat.replace('Ġ', '')
                dat = dat.replace('<|endoftext|>', '')
                dat = dat.replace('<|end_of_text|>', '') 
            data += [dat]

        return data
        