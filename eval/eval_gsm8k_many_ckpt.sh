device=cuda:7
data_dir=/xxx/eval/data/gsm8k
export HF_ENDPOINT=https://hf-mirror.com
python -u eval_gsm8k.py --checkpoint-path  /xxx/ChnAndEng_70wchineseinfinity_200wchoice_3wengchoice_codef20_openhrm20_webinst20_bz8/checkpoint-72000 --device $device