# Revisiting Commonsense Reasoning in Machine Translation: Training, Evaluation and Challenge

![Static Badge](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)
[![Static Badge](https://img.shields.io/badge/ACL%202023-long%20866-da3732?logoColor=red)](https://aclanthology.org/2023.acl-long.866/)


Code and data release for our [paper](https://aclanthology.org/2023.acl-long.866/) published in ACL 2023

<div align="center">
    <img src="image/Poster_ACL23_Revisiting Commonsense Reasoning_new.png" width="80%" title="Poster of the article"></img>
    <p class="image-caption">Poster of Our Article</p>
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

### Entity-Aware BERTScore
To calculate entity-aware BERTScore between candidates and right/contrastive references, run the following script 
(taking the candidates of Pt-Nmt as an example, model files will be downloaded automatically during initial running):
```bash
python ent_bertscore/ent_score.py \
    --candidate data/ptnmt/cr-test.hyp \
    --r_reference data/cr_testset/test.true.en \
    --c_reference data/cr_testset/test.wrong.en \
    --weight 1.4 \
    --outdir example/
```
More information about these parameters can be acquired by running `python ent_bertscore/ent_score.py --help`.

After running, two score files `true.score.bert_1.4` and `wrong.score.bert_1.4` will be found 
at the output directory `example/`, consisting of right side and contrastive side scores, respectively.


### Commonsense Reasoning Accuracy
Given the right and contrastive side scores, to calculate commonsense reasoning accuracy, run the following script
(results of the previous step are used here):
```bash
python CR_accuracy/calculate_accuracy.py \
    --r_score example/true.score.bert_1.4 \
    --c_score example/wrong.score.bert_1.4
```
More information about these parameters can be acquired by running `python CR_accuracy/calculate_accuracy.py --help`.


### Meta-Evaluation of Commonsense Reasoning Metrics
To conduct a meta-evaluation on the commonsense reasoning metrics, run the following scripts to examine the correlation 
between metric scores and human annotations (results of the first step are used here, but the scripts work 
not only for BERTScore, but also for other instance-level scores, such as Prob, BLEU and BLEURT):
```bash
# chi-square test
python meta-evaluation/evaluate.py \
    --r_score example/true.score.bert_1.4 \
    --c_score example/wrong.score.bert_1.4 \
    --human data/ptnmt/human.label \
    --method chi
    
# Kendall rank correlation coefficient
python meta-evaluation/evaluate.py \
    --r_score example/true.score.bert_1.4 \
    --c_score example/wrong.score.bert_1.4 \
    --human data/ptnmt/human.label \
    --method kendall
    
# Analysis of variance
python meta-evaluation/evaluate.py \
    --r_score example/true.score.bert_1.4 \
    --c_score example/wrong.score.bert_1.4 \
    --human data/ptnmt/human.label \
    --method anova
```
More information about these parameters can be acquired by running `python meta-evaluation/evaluate.py --help`.

## Citation
If you find this repo useful, please cite our [paper](https://aclanthology.org/2023.acl-long.866/):
```
@inproceedings{liu-etal-2023-revisiting-commonsense,
    title = "Revisiting Commonsense Reasoning in Machine Translation: Training, Evaluation and Challenge",
    author = "Liu, Xuebo  and
      Wang, Yutong  and
      Wong, Derek F.  and
      Zhan, Runzhe  and
      Yu, Liangxuan  and
      Zhang, Min",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.866",
    pages = "15536--15550",
    abstract = "The ability of commonsense reasoning (CR) decides whether a neural machine translation (NMT) model can move beyond pattern recognition. Despite the rapid advancement of NMT and the use of pretraining to enhance NMT models, research on CR in NMT is still in its infancy, leaving much to be explored in terms of effectively training NMT models with high CR abilities and devising accurate automatic evaluation metrics. This paper presents a comprehensive study aimed at expanding the understanding of CR in NMT.For the training, we confirm the effectiveness of incorporating pretrained knowledge into NMT models and subsequently utilizing these models as robust testbeds for investigating CR in NMT. For the evaluation, we propose a novel entity-aware evaluation method that takes into account both the NMT candidate and important entities in the candidate, which is more aligned with human judgement. Based on the strong testbed and evaluation methods, we identify challenges in training NMT models with high CR abilities and suggest directions for further unlabeled data utilization and model design. We hope that our methods and findings will contribute to advancing the research of CR in NMT. Source data, code and scripts are freely available at https://github.com/YutongWang1216/CR-NMT.",
}

```

## Menu
| Contents                                                     | Directory  |
|--------------------------------------------------------------| ----  |
| Experimental Data                                            | ./data/ |
| Chi-square test, Kendall's tau calculation and ANOVA scripts | ./evaluation/ |
| Entity-aware BERTScore                                       | ./ent_bertscore/ |
| CR accuracy calculation script                               | ./CR_accuracy/ |
