TOKENIZER_PATH="config/tokenizer_k6_1024"
DATASET="data/dataset.txt"

transforkmers pretrain \
	--tokenizer $TOKENIZER_PATH \
	--model_path_or_name big_bird \
	--dataset $DATASET \
	--output_dir models/pretrain_bb_8-512/ \
	--overwrite_output_dir True \
	--per_device_train_batch_size 14 \
	--gradient_accumulation_steps 4 \
	--save_strategy steps \
	--max_steps 200000 \
	--warmup_steps 10000 \
	--save_steps 5000 \
	--logging_steps 50 \
	--save_total_limit 5 \
	--prediction_loss_only True \
	--learning_rate 1e-4 \
	--save_total_limit 10 \
	--disable_tqdm True \
	--report_to all \
	--fp16 \
	--overload_config "hidden_size=516,num_hidden_layers=8,intermediate_size=1536"
