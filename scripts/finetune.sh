#!/bin/bash
#SBATCH --gres=gpu:1 -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

###############################################################################
MAX_LEN="1024"
TOKENIZER_PATH="/scratch/mlorthiois/test_bert/config/tokenizer_k6_$MAX_LEN"
MODEL_PATH="/scratch/mlorthiois/test_bert/models/pretrain_bb_8-512/checkpoint-30000"
DATASET_OUTDIR="/scratch/mlorthiois/test_bert/data/03_processed/1024/hsa.5prime"

###############################################################################
. /local/env/envconda3.sh
conda activate ../envs

#transforkmers finetune-dataset \
	#--inputs \
		#/scratch/mlorthiois/test_bert/data/02_intermediate/1024/hsa.false.species.5prime.fa \
		#/scratch/mlorthiois/test_bert/data/02_intermediate/1024/Homo_sapiens.5prime.fa \
	#--split 75,10,15 \
	#--output_dir $DATASET_OUTDIR

transforkmers finetune \
	--tokenizer $TOKENIZER_PATH \
	--model_path_or_name $MODEL_PATH \
	--train_dataset "${DATASET_OUTDIR}_train.csv" \
	--eval_dataset "${DATASET_OUTDIR}_eval.csv" \
	--num_labels 2 \
	--output_dir "/scratch/mlorthiois/test_bert/models/hsa_5prime_bb_8-512_1024-$MAX_LEN/" \
	--evaluation_strategy steps \
	--save_strategy steps \
	--eval_steps 5000 \
	--save_steps 5000 \
	--max_steps 100000 \
	--learning_rate 5e-7 \
	--per_device_train_batch_size 15 \
	--per_device_eval_batch_size 24 \
	--save_total_limit 10 \
	--optim adamw_torch \
	--load_best_model_at_end \
	--metric_for_best_model loss \
	--disable_tqdm True \
	--report_to all \
	--fp16 \
	--patience 5
