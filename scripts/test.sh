#!/bin/bash
#SBATCH --gres=gpu:1 -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

###############################################################################
MAX_LEN="512"
TOKENIZER_PATH="/scratch/mlorthiois/test_bert/config/tokenizer_k6_$MAX_LEN"
MODEL_PATH="/scratch/mlorthiois/test_bert/models/hsa_5prime_bert_6_12-512/best"
TEST_DATASET="/scratch/mlorthiois/test_bert/data/03_processed/512/hsa_5prime_test.csv"

###############################################################################
. /local/env/envconda3.sh
conda activate ../envs

transforkmers test \
	--output_dir "test_hsa_only" \
	--tokenizer $TOKENIZER_PATH \
	--model_path_or_name $MODEL_PATH \
	--test_dataset $TEST_DATASET \
	--per_device_eval_batch_size 24 \
	--disable_tqdm True \
	--report_to all
