# ENLP-BabyLM

This project is built upon the [Stanford CRFM Mistral framework](https://github.com/stanford-crfm/mistral/tree/main) [Injecting structural hints: Using language models to study inductive biases in language learning](https://github.com/toizzy/injecting-structural-hints/tree/main)and focuses on pretraining and finetuning GPT2-small.

## Environment Setup

Please follow [Mistral repository](https://github.com/stanford-crfm/mistral/tree/main) installation guide for environment setup first.

Install required packages for finetuning:

```bash
pip install ray[tune]==1.13.0
pip install datasets==1.18.4
pip install numpy==1.23.5
```

## Pretraining

### Generate Synthetic Data

Navigate to the data generation script folder and generate the synthetic dataset:

```bash
cd synthetic_corpora/corpus_creation_scripts/nested_parens
python nested_parens.py --open-prob 0.49 --vocab-size 500 --vocab-distribution uniform
```

### Run Pretraining

Launch pretraining with `torchrun`:

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 train.py \
  --config conf/inductive-bias-pretraining.yaml \
  --training_arguments.fp16 true \
  --training_arguments.per_device_train_batch_size 64 \
  --dataset.name nested-parens0.49_vocab500-uniform \
  --run_id nested-parens0.49_vocab500-uniform
```

## Finetuning

### Dataset Preparation

Download the dataset from [OSF Storage](https://osf.io/ad7qg/files/osfstorage), and extract the three `.zip` files into:

```
ENLP-BabyLM/babylm_dataset/
```

### Run Finetuning

Finetuning depends on the checkpoint generated during pretraining (default: step 5000).  
If you modified the pretraining steps, update the checkpoint path in `finetune.py` (line 49).

Run:

```bash
python finetune.py --config-file conf/sample.yaml
```