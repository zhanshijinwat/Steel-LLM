# https://huggingface.co/docs/huggingface_hub/main/en/guides/cli
export HF_ENDPOINT=https://hf-mirror.com
# 需要先安装aria2c和git-lfs
# https://hf-mirror.com/往下拉有下载教程
../sky/hfd.sh BelleGroup/train_3.5M_CN  --dataset --tool aria2c -x 16 --local-dir /data1/step0_rawdata/BELLE/3_5M
../sky/hfd.sh BelleGroup/train_2M_CN  --dataset --tool aria2c -x 16 --local-dir /data1/step0_rawdata/BELLE/2M
