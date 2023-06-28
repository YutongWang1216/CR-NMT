import argparse


def calculate_accuracy():
    right = open(rscore, 'r').readlines()
    contrastive = open(cscore, 'r').readlines()

    true_score = [float(i.replace("\n", "")) for i in right if i != "\n"]
    wrong_score = [float(i.replace("\n", "")) for i in contrastive if i != "\n"]

    assert len(true_score) == 1200
    assert len(wrong_score) == 1200

    subset_list = ['CL_SA', 'CT_SA', 'LA', 'ALL']

    for idx, bound in enumerate([(0, 450), (450, 800), (800, 1200), (0, 1200)]):
        print(subset_list[idx])
        begin, end = bound
        cnt = 0
        for ts, ws in zip(true_score[bound[0]:bound[1]], wrong_score[bound[0]:bound[1]]):
            if ts > ws:
                cnt = cnt + 1

        print("%.1f%%" % (cnt / (end - begin) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate entity-aware BERTScore.')
    parser.add_argument('-r', '--r_score', type=str, required=True, help='Right side score file')
    parser.add_argument('-c', '--c_score', type=str, required=True, help='Contrastive side score file')

    args = parser.parse_args()
    rscore = args.r_score
    cscore = args.c_score

    calculate_accuracy()
