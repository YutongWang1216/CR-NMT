import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd
from scipy.stats import kendalltau
from sklearn.feature_selection import f_classif
import argparse


def calc_score(right_score, contranstive_score, human_labels):
    if args.method == 'chi':  # 卡方检验
        metric_labels = [1 if i > j else 0 for (i, j) in zip(right_score, contranstive_score)]
        table = [[0, 0], [0, 0]]
        for i, j in zip(metric_labels, human_labels):
            table[i][j] = table[i][j] + 1
        df = pd.DataFrame(table, index=['truescore<=wrongscore', 'truescore>wrongscore'], columns=['wrong', 'true'])
        kt = chi2_contingency(df, correction=False)
        print('%.1f\t%e' % (kt[0], kt[1]))
    else:
        metric_score = [i - j for (i, j) in zip(right_score, contranstive_score)]
        if args.method == 'kendall':  # Kendall 秩相关系数
            f, p = kendalltau(metric_score, human_labels)
            print('%.3f\t%e' % (f, p))
        else:  # ANOVA 方差分析
            matric_score = [[i - j] for (i, j) in zip(right_score, contranstive_score)]
            f, p = f_classif(matric_score, human_labels)
            print('%.3f\t%e' % (f, p))


def evaluate():

    right_file = open(args.rscore, 'r').readlines()
    contrastive_file = open(args.cscore, 'r').readlines()
    human_file = open(args.human, 'r').readlines()
    
    right_score = np.array([float(i.replace("\n", "")) for i in right_file if i != "\n"])
    contranstive_score = np.array([float(i.replace("\n", "")) for i in contrastive_file if i != "\n"])
    human_labels = [int(i.replace("\n", "")) for i in human_file if i != "\n"]

    assert len(right_score) == 1200
    assert len(contranstive_score) == 1200

    subset_list = ['CL_SA', 'CT_SA', 'LA', 'ALL']

    for idx, bound in enumerate([(0, 450), (450, 800), (800, 1200), (0, 1200)]):
        print(subset_list[idx])
        begin, end = bound
        calc_score(right_score[begin:end], contranstive_score[begin:end], human_labels[begin:end])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-evaluate the metrics.')
    parser.add_argument('rscore', type=str, help='Path to right side scores')
    parser.add_argument('cscore', type=str, help='Path to contrastive side scores')
    parser.add_argument('human', type=str, help='Path to human labels')
    parser.add_argument('method', type=str, choices=['chi', 'kendall', 'anova'], help='Select an evaluation method')

    args = parser.parse_args()

    evaluate()
