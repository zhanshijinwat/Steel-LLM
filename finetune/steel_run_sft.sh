##########################
# llama factory start script
##########################
#export OMP_NUM_THREADS=1
# python -X faulthandler src/train_bash.py
#CUDA_VISIBLE_DEVICES=5,6 python ../LLaMA-Factory/src/train.py \
export LLaMA_PATH=/home/LLaMA-Factory
OUTPUT_DIR=/data/model/llm/fintuned_model
# CUDA_VISIBLE_DEVICES=6,7 python  $LLaMA_PATH/src/train.py \
# max_steps / num_train_epochs
# --streaming True \
# /data/model//steel-llm-step-1060000-ckpt
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node 6 $LLaMA_PATH/src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path /data/model/steel-llm-step-1060000-ckpt \
    --cutoff_len 768 \
    --dataset_dir $LLaMA_PATH/data \
    --dataset  ruozhiba,infinity_chinese,wanjuan_exam_390w,self_cog \
    --preprocessing_num_workers 72 \
    --template qwen \
    --finetuning_type full \
    --output_dir ${OUTPUT_DIR}/from_pretrain_70wchineseinfinity_390wchoice_selfcog_ruozhiba_bz8 \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 2000 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16
