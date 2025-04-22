#!/bin/bash

echo "Evaluating random..."
python eval.py --config-file conf/eval.yaml --model-name finetune-random_vocab500-uniform_seed8 --checkpoint-number 1500                                                           

echo "Evaluating nested"
python eval.py --config-file conf/eval.yaml --model-name finetune-mixed-parens0.005_vocab500-uniform_seed8 --checkpoint-number 1500                                                           

echo "Evaluating mixed "
python eval.py --config-file conf/eval.yaml --model-name finetune-nested-parens0.49_vocab500-uniform_seed8 --checkpoint-number 1500                                                           

echo "Evaluating dependency "
python eval.py --config-file conf/eval.yaml --model-name finetune-constituency_seed8 --checkpoint-number 1500                                                           

echo "Evaluating constituency "
python eval.py --config-file conf/eval.yaml --model-name finetune-dependency_seed8 --checkpoint-number 1500                                                           

echo "All tasks done!"
