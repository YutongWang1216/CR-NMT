# Revisiting Commonsense Reasoning in Machine Translation:Training, Evaluation and Challenge

Implementation of our paper published in ACL 2023


<div align="center">
    <img src="IMG_00001.png">
</div>


The ability of commonsense reasoning (CR)
decides whether a neural machine translation
(NMT) model can move beyond pattern recognition. Despite the rapid advancement of NMT
and the use of pretraining to enhance NMT
models, research on CR in NMT is still in its infancy, leaving much to be explored in terms of
effectively training NMT models with high CR
abilities and devising accurate automatic evaluation metrics. This paper presents a comprehensive study aimed at expanding the understanding of CR in NMT. For the training, we confirm
the effectiveness of incorporating pretrained
knowledge into NMT models and subsequently
utilizing these models as robust testbeds for
investigating CR in NMT. For the evaluation,
we propose a novel entity-aware evaluation
method that takes into account both the NMT
candidate and important entities in the candidate, which is more aligned with human judgement. Based on the strong testbed and evaluation methods, we identify challenges in training NMT models with high CR abilities and
suggest directions for further unlabeled data
utilization and model design. We hope that our
methods and findings will contribute to advancing the research of CR in NMT.


## Installation
```bash
git clone https://github.com/YutongWang1216/CR-NMT.git
cd CR-NMT
pip install -r requirements.txt
```


## Examples
To calculate entity-aware BERTScore between candidates and right/contrastive references, run this script 
(taking the candidates of Pt-Nmt as an example):
```bash
python ent_bertscore/ent_score.py \
    --candidate data/ptnmt/cr-test.hyp \
    --r_reference data/cr_testset/test.true.en \
    --c_reference data/cr_testset/test.wrong.en \
    --weight 1.4 \
    --outdir example/
```
More information about these parameters can be found by running `python ent_bertscore/ent_score.py --help`.

After running, two score files `true.score.bert_1.4` and `wrong.score.bert_1.4` will be found 
at the output directory `example/`, consisting of right side and contrastive side scores, respectively.

## 
```bash

```

# Menu

|  Contents   | Directory  |
|  ----  | ----  |
| Data  | ./data/ |
| Chi-square test, Kendall's tau calculation and ANOVA scripts  | ./evaluation/ |
| Entity-aware BERTScore  | ./ent_bertscore/ |
| CR accuracy calculation script  | ./CR_accuracy/ |
