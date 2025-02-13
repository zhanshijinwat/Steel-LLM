data_dir=/xxx/eval/data/mmlu
device=cuda:0
python -u eval_mmlu.py --checkpoint-path /xxx/ChnAndEng_70wchineseinfinity_200wchoice_3wengchoice_codef20_openhrm20_webinst20_bz8/checkpoint-24000 --eval_data_path $data_dir --device $device