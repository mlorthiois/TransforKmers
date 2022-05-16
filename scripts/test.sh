#!/bin/bash
#SBATCH --gres=gpu:1 -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

###############################################################################
MAX_LEN="1024"
TOKENIZER_PATH="/scratch/mlorthiois/test_bert/config/tokenizer_k6_$MAX_LEN"
MODEL_PATH="/scratch/mlorthiois/test_bert/models/hsa_5prime_bb_8-512_1024/best"
TEST_DATASET="/scratch/mlorthiois/test_bert/data/03_processed/$MAX_LEN/hsa.5prime_test.csv"

###############################################################################
. /local/env/envconda3.sh
conda activate ../envs

transforkmers test \
	--output_dir "../results/test/hsa/1024/bb_8-512_1024" \
	--tokenizer $TOKENIZER_PATH \
	--model_path_or_name $MODEL_PATH \
	--test_dataset $TEST_DATASET \
	--per_device_eval_batch_size 24 \
	--disable_tqdm True \
	--report_to all
