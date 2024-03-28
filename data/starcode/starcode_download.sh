# https://huggingface.co/docs/huggingface_hub/main/en/guides/cli
export HF_ENDPOINT=https://hf-mirror.com
# 需要先安装aria2c和git-lfs
# https://hf-mirror.com/往下拉有下载教程
../sky/hfd.sh bigcode/starcoderdata --dataset --tool aria2c -x 16 --local-dir /data/starcode --include java/*parquet --include java/*parquet --include python/*parquet --hf_token *** --hf_username ***
