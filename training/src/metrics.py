import torch

from collections import Counter
from typing import List
from transformers import T5Tokenizer

def perplexity(batch, model, tokenizer):
    # all labels set to -100 are ignored; the loss is only calculated 
    # for the other tokens.
    labels = batch['target'].clone()
    labels[batch['target'][:, :] == tokenizer.pad_token_id] = -100
    with torch.inference_mode():
        outputs = model(input_ids=batch['input'].squeeze(), 
            labels=labels.squeeze())
        loss = outputs.loss        
        return round(torch.exp(loss).item(), 3)
    
def n_gram_overlapping_f1(preds:List[str], 
                          golds:List[str], 
                          tokenizer:T5Tokenizer):
    """
    This method uses the number of overlapping token
    between predictions and golds to calculate precision,
    recall, and F1.
    
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    prec, rec, f1 = [], [], []

    preds = [tokenizer.tokenize(text) for text in preds]
    golds = [tokenizer.tokenize(text) for text in golds]

    for pred, gold in zip(preds, golds):
        common = Counter(gold) & Counter(pred) 
        num_same = sum(common.values())
        if num_same == 0:
            prec += [0]
            rec += [0]
            f1 += [0]
            continue
        prec += [1.0 * num_same / len(pred)]
        rec += [1.0 * num_same / len(gold)]
        f1 += [(2 * prec[-1] * rec[-1]) / (prec[-1] + rec[-1])]

    return round(sum(prec)/len(prec), 3),\
        round(sum(rec)/len(rec), 3),\
        round(sum(f1)/len(f1), 3)