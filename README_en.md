<div align="center">

# Open Source Chinese Pre-trained Language Model Steel-LLM
Created by <a href="https://www.zhihu.com/people/zhan-shi-jin-27">zhanshijin</a> and <a href="https://www.zhihu.com/people/a-xun-58-5">lishu14</a>
</div>

\[ [中文](README.md) | English \]

## 👋 Introduction
Steel-LLM is a personally initiated project to pre-train a large Chinese model from scratch. We pre-trained a Chinese LLM with approximately 1 billion parameters using 1 trillion tokens of data. The project took 8 months from initiation to the completion of the first version of the model. We have shared the entire process in detail, including data collection, data processing, pre-training framework selection, and model design, and have open-sourced all the code. This enables anyone with 8 to several dozen GPUs to reproduce our work. Thanks to open-source Chinese data, Steel-LLM outperforms some larger early models released by institutions on Chinese benchmarks, scoring 38 in CEVAL and 33 in CMMLU.
<div align="center">
  <img src=".github/steel.png" width="200"/>
</div>

<p align="center">
        <a href="https://huggingface.co/gqszhanshijin/Steel-LLM" target="_blank">
        <img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97Hugging%20Face-Steel%20LLM-yellow">
        </a>&nbsp&nbsp
        <a href="https://www.modelscope.cn/models/zhanshijin/Steel-LLM"><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%96modelscope-Steel%20LLM-purple">
        </a>&nbsp&nbsp
        <a href=".github/wechat_liangangAI.jpg">
        <img alt="Static Badge" src="https://img.shields.io/badge/WeChat-%E7%82%BC%E9%92%A2AI-brightgreen?logo=wechat&logoColor=white">
        </a>&nbsp&nbsp
        <a href="https://www.zhihu.com/people/zhan-shi-jin-27">
        <img alt="Static Badge" src="https://img.shields.io/badge/%E2%98%91%EF%B8%8Fzhihu%20blog-%E6%88%98%E5%A3%AB%E9%87%91-blue">
        </a>

"Steel" was inspired by an excellent band from the North China Plain called “Universal Youth Hostel (万青)”. When they were producing their first album under limited conditions, they referred to it as "making steel with primitive methods," but it turned out to be a legendary album. Similarly, our conditions for training the LLM are limited, but we hope to produce good "steel" nonetheless.

## 🔔 Announcements

### Updates
[2024/11/13] Based on Steel-LLM, further SFT optimization (mainly focusing on SFT sample selection) and reinforcement learning training.

[2024/11/13] 🔥 Published a project summary article "My Journey of Pre-training a 1B LLM from Scratch": https://mp.weixin.qq.com/s/POUugkCNZTzmlKWZVVD1CQ 🔥 , technical report in preparation...

[2024/10/28] Updated the first version of the chat model, scoring 38 in CEVAL and 33 in CMMLU.

[2024/10/24] Published the details of Steel-LLM fine-tuning and evaluation. During fine-tuning, we explored experiments such as CoT and model leaderboard. Blog address: https://mp.weixin.qq.com/s/KK0G0spNw0D9rPUESkHMew

[2024/9/2] HuggingFace updated the checkpoints for 480k, 660k, 720k, 980k, 1060k (the final checkpoint) steps.

[2024/8/18] Pre-training completed, followed by fine-tuning and evaluation.

[2024/7/18] Continued training using 8*H800, wandb: https://api.wandb.ai/links/steel-llm-lab/vqf297nr

[2024/6/30] Released the checkpoint for 200k steps of pre-training, [HuggingFace link](https://huggingface.co/gqszhanshijin/Steel-LLM/tree/main)

[2024/5/21] Official training of the model began, and checkpoints will be released periodically.

[2024/5/19] Completed model modification based on Qwen1.5, model size 1.12B:
- FFN layer uses softmax moe, providing higher training speed with the same parameter count
- Uses dual-layer SwiGLU

Related blog: https://zhuanlan.zhihu.com/p/700395878

[2024/5/5] Blog on modifying the pre-training program: https://zhuanlan.zhihu.com/p/694223107

[2024/4/24] Completed improvements to the training program: compatible with HuggingFace format models, support for data checkpointing, support for adding new data.

[2024/4/14] Completed data collection and processing, generating bin files required for the pre-training program. Updated blog on data collection and processing: https://zhuanlan.zhihu.com/p/687338497

### 🧑‍🤝‍🧑 Community
You are welcome to join our community group. The number of members has exceeded 200. Add WeChat to join the group: a1843450905

<br>

## 🤖 Pre-training
### Data Collection
The datasets used and their links are listed below. For more details, see [**this article**](https://zhuanlan.zhihu.com/p/687338497)

- [Skywork/Skypile-150B data](https://huggingface.co/datasets/Skywork/SkyPile-150B/tree/main/data)
- [wanjuan1.0 (NLP part)](https://opendatalab.org.cn/OpenDataLab/WanJuan1_dot_0?source=Q1NETg)
- [Filtered Chinese Wikipedia data](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- [Baidu Encyclopedia data](https://huggingface.co/datasets/xuqinyang/BaiduBaike-5.63M)
- [Baidu Encyclopedia Q&A data](https://aistudio.baidu.com/datasetdetail/107726)
- [Zhihu Q&A data](https://huggingface.co/datasets/wangrui6/Zhihu-KOL)
- [BELLE dialogue data](https://github.com/LianjiaTech/BELLE/tree/main/data/10M)
- [MOSS project dialogue data](https://hf-mirror.com/datasets/YeungNLP/moss-003-sft-data)
- [firefly1.1M](https://hf-mirror.com/datasets/YeungNLP/firefly-train-1.1M)
- [starcoder](https://hf-mirror.com/datasets/bigcode/starcoderdata)

### Data Processing
(For details, see [**this article**](https://mp.weixin.qq.com/s/yqmtHLuuNV9075qHgzhcPw))
#### Step 1: Format Conversion
- Source data: Unified format processing for 4 types of data:
  - Simple text: Baidu Encyclopedia (manual merging of title and paragraphs), Chinese Wikipedia
  - Dialogue (including single and multi-round): Baidu Encyclopedia Q&A data, BELLE dialogue data (BELLE_3_5M), MOSS project dialogue data, Zhihu Q&A data, BELLE task data (BELLE_2_5M), firefly1.1M
  - Code data: starcoder
  - Other data: No separate processing required for wanjuan and skypile datasets
- Target format: `{"text": "asdfasdf..."}`, file saved as .jsonl format.
- Run command: `python data/pretrain_data_prepare/step1_data_process.py`
#### Step 2: Data Juicer Processing
- Run command: `sh data/pretrain_data_prepare/step2/run_step2.sh`
- Specific data juicer operators used are documented in <a href=".github/datajuicer_op_en.md">this document</a>.

#### Step 3: Generating Final Training Bin Format
Modify `filename_sets` in the code to specify the data path, then run the following program:

`python pretrain_modify_from_TinyLlama/scripts/prepare_steel_llm_data.py`

Input data format: jsonl file containing the 'text' field

### Tokenizer
We do not train a tokenizer separately and use the tokenizer from [Qwen/Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)

### Model Architecture
(For details, see [**this article**](https://mp.weixin.qq.com/s/JaZyf1jOEOtNDCcFqSj8TQ))

Based on the Qwen1.5 model, the following changes were made:
- FFN layer uses softmax moe, providing higher training speed with the same parameter count
- Uses dual-layer SwiGLU

### Pre-training Framework
(For details, see [**this article**](https://mp.weixin.qq.com/s/KPRir6bK3MZZ-vMFTfhUQQ))

Based on the TinyLlama pre-training program, the following improvements were made:
  - Compatible with HuggingFace format models
  - Fully restores training progress when loading checkpoints
  - Data consistency checks
  - Adds new data to the dataset without affecting already trained data

Start pre-training:

`python Steel-LLM/pretrain_modify_from_TinyLlama/pretrain/pretrain_steel_llm.py`

### Evaluation
(For details, see [**this article**](https://mp.weixin.qq.com/s/KK0G0spNw0D9rPUESkHMew))

Steel-LLM was tested on CEVAL and CMMLU. Steel-LLM aims to train a Chinese LLM, with 80% of the training data being Chinese, so it was not evaluated on English benchmarks.
Metrics for other models come from the CEVAL paper, MiniCPM technical report, MAP-Neo technical report, etc. More model metrics can be found in the previous <a href=https://mp.weixin.qq.com/s/KK0G0spNw0D9rPUESkHMew>blog</a>.

|                              | CEVAL  | CMMLU |
|------------------------------|--------|-------|
| Steel-LLM                    | 38.57  | 33.48 |
| Tiny-Llama-1.1B              | 25.02  | 24.03 |
| Gemma-2b-it                  | 32.3   | 33.07 |
| Phi2(2B)                      | 23.37  | 24.18 |
| Deepseek-coder-1.3B-instruct |  28.33 | 27.75 |
| CT-LLM-SFT-2B                | 41.54  | 41.48 |
| MiniCPM-2B-sft-fp32          | 49.14  | 51.0  |
| Qwen1.5-1.8B-Chat            | 56.84  | 54.11 |
| ChatGLM-6B                   | 38.9   | -     |
| Moss                         | 33.1   | -     |
| LLAMA-65B                    | 34.7   | -     |
| Qwen-7B                      | 58.96  | 60.35 |
| Gemma-7B                     | 42.57  | 44.20 |
| OLMo-7B                      | 35.18  | 35.55 |
| MAP-NEO-7B                   | 56.97  | 55.01 |

<br>

## ⛏️ Quick Usage
```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "zhanshijin/Steel-LLM"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Who developed you?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

```

### Hardware Resources
GPU: 8* H800 80G (training for about 30 days)

GPU: 8* A100 80G (training for about 60 days)

Disk: 4TB
