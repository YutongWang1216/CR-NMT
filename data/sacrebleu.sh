testfile=$1
sacrebleu -l zh-en -t wmt20 < $testfile
