from commonsence_score import cs_score
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer


def calc_score():
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=False)

    with open(args.cand, 'r') as f:
        cands = [line.strip() for line in f]
    with open(args.rref, 'r', encoding="utf-8") as f:
        right_refs = [line.strip() for line in f]
    with open(args.cref, 'r', encoding="utf-8") as f:
        contrastive_refs = [line.strip() for line in f]

    print('Current weight of commonsense entities: %f' % (args.weight))

    scores = cs_score(cands, right_refs, contrastive_refs, lang='en', verbose=True, 
                    ent_weight=args.weight, tokenizer=tokenizer, idf=False, device="cpu" if args.cpu else None)
    right_f = np.array(scores[0][2])
    contrastive_f = np.array(scores[1][2])

    return right_f, contrastive_f


def write_score(right_score, contrastive_score):
    right_path = args.outdir + "/true.score.bert_" + str(args.weight)
    contrastive_path = args.outdir + "/wrong.score.bert_" + str(args.weight)

    with open(right_path, 'w') as f:
        for i in right_score:
            f.write('%.6f\n' % (i))

    with open(contrastive_path, 'w') as f:
        for i in contrastive_score:
            f.write('%.6f\n' % (i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate entity-aware BERTScore.')
    parser.add_argument('cand', type=str, help='Path to candidate')
    parser.add_argument('rref', type=str, help='Path to right reference')
    parser.add_argument('cref', type=str, help='Path to contrastive reference')
    parser.add_argument('outdir', type=str, help='Path to output directory')
    parser.add_argument('--weight', '-w', type=float, default=1.4, help='Weight of commonsense entities')
    parser.add_argument('--cpu', action="store_true", help='Run on cpu')

    args = parser.parse_args()

    rf, cf = calc_score()
    write_score(rf, cf)
