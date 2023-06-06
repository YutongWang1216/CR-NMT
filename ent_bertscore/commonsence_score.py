from transformers import AutoTokenizer
import torch
from score import score
from collections import Counter, defaultdict
from functools import partial
from itertools import chain
from multiprocessing import Pool
# from utils import sent_encode, process
from utils_ent_fewer import sent_encode, process


def cs_score(cands, correct_refs, wrong_refs, lang, verbose, device=None, ent_weight=1.4, idf=False, tokenizer=None):
    correct_mask_dict = None
    wrong_mask_dict = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=False)
    
    
    if ent_weight != 1.0:
        
        correct_mask_dict = dict()
        wrong_mask_dict = dict()
    
        for correct_ref, wrong_ref in zip(correct_refs, wrong_refs):
            correct_tokens = tokenizer.encode(correct_ref, add_special_tokens=True, truncation=True)
            wrong_tokens = tokenizer.encode(wrong_ref, add_special_tokens=True, truncation=True)

            # 提取常识实体
            correct_set = set(correct_tokens)
            wrong_set = set(wrong_tokens)

            correct_ents = correct_set - wrong_set
            wrong_ents = wrong_set - correct_set

            # 为常识实体创建 mask
            correct_mask = torch.tensor([1 if i in correct_ents else 0 for i in correct_tokens])
            wrong_mask = torch.tensor([1 if i in wrong_ents else 0 for i in wrong_tokens])

            correct_mask_dict[correct_ref] = correct_mask
            wrong_mask_dict[wrong_ref] = wrong_mask

    # 计算常识实体感知的 BERTScore, 传入每个参考依据的常识实体 mask
    correct_scores = score(cands, correct_refs, lang=lang, verbose=verbose, idf=idf, device=device, ent_dict=correct_mask_dict, ent_weight=ent_weight)
    wrong_scores = score(cands, wrong_refs, lang=lang, verbose=verbose, idf=idf, device=device, ent_dict=wrong_mask_dict, ent_weight=ent_weight)

    return correct_scores, wrong_scores
