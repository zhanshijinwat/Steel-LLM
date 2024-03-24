# https://huggingface.co/docs/huggingface_hub/main/en/guides/cli
export HF_ENDPOINT=https://hf-mirror.com
# 方法1：
# 断了没法继续下载
# huggingface-cli download Skywork/SkyPile-150B --repo-type dataset --resume-download --local-dir /hoe/test/gqs/Steel-LLM/data/sky  --local-dir-use-symlinks False
# 方法2
# 需要先安装aria2c和git-lfs
# https://hf-mirror.com/往下拉有下载教程
./hfd.sh Skywork/SkyPile-150B --dataset --tool aria2c -x 16 --local-dir /hoe/test/gqs/Steel-LLM/data/sky/data2