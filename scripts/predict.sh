#!/bin/bash
#SBATCH --gres=gpu:1 -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

###############################################################################
MAX_LEN="512"
TOKENIZER_PATH="/scratch/mlorthiois/test_bert/config/tokenizer_k6_$MAX_LEN"
MODEL_PATH="/scratch/mlorthiois/test_bert/models/hsa_5prime_bert_6_12-512/best"
PREDICT_DATASET="/groups/dog/nanopore/lncrna_resist_cgo/secondary/annexa/results/feelnc/feelnc.combined.gtf"
FASTA="/groups/dog/data/hg38_GRCh38/sequence/softmasked/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"
PREFIX="lncrna_resist_hsa"

###############################################################################
. /local/env/envconda3.sh
mkdir -p $PREFIX

echo "Extracting tts regions"
chmod +x /scratch/mlorthiois/test_bert/scripts/extract_regions.py
/scratch/mlorthiois/test_bert/scripts/extract_regions.py -i $PREDICT_DATASET -l $MAX_LEN -r 5 > "$PREFIX/tss.bed"


. /local/env/envbedtools-2.27.1.sh
echo "Extracting tts sequences"
bedtools getfasta -name -fi $FASTA -bed $PREFIX/tss.bed | tr a-z A-Z > $PREFIX/tss.fa


echo "Predicting..."
conda activate /scratch/mlorthiois/test_bert/envs/

transforkmers predict \
	--output_dir $PREFIX \
	--tokenizer $TOKENIZER_PATH \
	--model_path_or_name $MODEL_PATH \
	--input $PREFIX/tss.fa \
	--per_device_eval_batch_size 24 \
