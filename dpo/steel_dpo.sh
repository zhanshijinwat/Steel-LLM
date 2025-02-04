export LLaMA_PATH=../
OUTPUT_DIR=/data/model/llm/dpo_model
# max_steps / num_train_epochs
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node 4 $LLaMA_PATH/src/train.py \
    --stage dpo \
    --do_train \
    --pref_beta 0.1 \
    --pref_loss sigmoid \
    --model_name_or_path /data/model/llm/fintuned_model/ChnAndEng_70wchineseinfinity_200wchoice_3wengchoice_codef20_openhrm20_webinst20_bz8/checkpoint-72000 \
    --cutoff_len 1536 \
    --dataset_dir $LLaMA_PATH/data \
    --dataset dpo_ultrafeedback_chinese_and_english \
    --preprocessing_num_workers 72 \
    --template qwen \
    --finetuning_type full \
    --output_dir ${OUTPUT_DIR}/dpo_ultrafeedback_chinese_and_20%english \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16