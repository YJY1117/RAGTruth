CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2 train.py \
--model_name_or_path /root/autodl-tmp/Llama-3.2-3B \
--output_dir /root/autodl-tmp/exp/baseline \
--do_train \
--dataset_config detect_yesno \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--drop_neg_ratio -1 \
--train_file ./train.jsonl \
--eval_file ./dev.jsonl \
--bf16 True \
--tf32 True \
--use_flashatt_2 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--model_max_length 4096 \
--report_to wandb \
--ddp_find_unused_parameters False \
--logging_steps 1 \
--run_name baseline \
--lr_scheduler_type 'cosine' \
--warmup_ratio 0.1 \
--save_steps 10000 \
--save_total_limit 2 \
--overwrite_output_dir \
--eval_strategy steps \
--eval_steps 80 \
--fsdp "shard_grad_op auto_wrap" \
--fsdp_config ./configs/fsdp.json