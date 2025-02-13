##########################
# llama factory start script
##########################
#CUDA_VISIBLE_DEVICES=5,6 python ../LLaMA-Factory/src/train.py \
export LLaMA_PATH=/xxx/LLaMA-Factory
OUTPUT_DIR=/xxx/fintuned_model
# CUDA_VISIBLE_DEVICES=6,7 python  $LLaMA_PATH/src/train.py \
# max_steps / num_train_epochs
# --streaming True \
# 6卡配置：1,2,3,4,5,7
# 原始ckpt：--model_name_or_path /data/model/llm/Steel-LLM/steel-llm-step-1060000-ckpt \
# all data: 
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2 $LLaMA_PATH/src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path /xxx/Steel-LLM/steel-llm-step-1060000-ckpt \
    --cutoff_len 768 \
    --dataset_dir $LLaMA_PATH/data \
    --dataset ruozhiba,infinity_chinese,wanjuan_exam,self_cog,code_feedback_random20,openhermes_random20,webinstruct_english_random20,english_choice \
    --preprocessing_num_workers 72 \
    --template qwen \
    --finetuning_type full \
    --output_dir ${OUTPUT_DIR}/ChnAndEng_70wchineseinfinity_200wchoice_3wengchoice_codef20_openhrm20_webinst20_bz8 \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 3000 \
    --learning_rate 3e-5 \
    --weight_decay 0.0 \
    --num_train_epochs 5.0 \
    --plot_loss \
    --bf16
