<div align="center">

# 开源中文预训练语言模型Steel-LLM
</div>

\[ 中文 | [English](README_en.md) \]

## 👋 介绍
Steel-LLM是个人发起的从零预训练中文大模型项目。我们使用了1T token的数据预训练一个1B左右参数量的中文LLM。项目从开始到微调出第一版模型耗时了8个月。我们详细的分享了数据收集、数据处理、预训练框架选择、模型设计等全过程，并开源全部代码。让每个人在有8~几十张卡的情况下都能复现我们的工作。得益于开源中文数据，Steel-LLM在中文benchmark上表现优于机构早期发布的一些更大的模型，在ceval达到了42分，cmmlu达到了36分。
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
        </a>&nbsp&nbsp
        <a href="https://arxiv.org/abs/2502.06635" target="_blank">
        <img alt="Arxiv" src="https://img.shields.io/badge/Arxiv-Steel%20LLM-7289da?logo=arxiv&logoColor=red&color=red" />
        </a>


        

"Steel(钢)"取名灵感来源于华北平原一只优秀的乐队“万能青年旅店（万青）”。乐队在做一专的时候条件有限，自称是在“土法炼钢”，但却是一张神专。我们训练LLM的条件同样有限，但也希望能炼出好“钢”来。

## 🔔 公告 

### 更新
后续会在数学能力、强化学习、复杂推理等方面进一步探索......

[2025/2/13] 上传了技术报告：https://arxiv.org/abs/2502.06635

[2025/1/17]  更新steel-LLM-chat-v2,微调时加入了英文数据，中英文数据比例和预训练保持一致，最终在ceval上由38分提高到了41.9分，cmmlu从33分提高到了36分。

[2024/11/13] 🔥发布一篇项目汇总文章《个人从零预训练1B LLM心路历程》：https://mp.weixin.qq.com/s/POUugkCNZTzmlKWZVVD1CQ🔥 

[2024/10/28]更新了第一版chat模型，在ceval达到了38分，cmmlu达到了33分。

[2024/10/24]发布了Steel-LLM微调和评估的细节。微调时探索了cot、模型刷榜等实验。博客地址：https://mp.weixin.qq.com/s/KK0G0spNw0D9rPUESkHMew

[2024/9/2] HuggingFace更新了480k、660k、720k、980k、1060k（最后一个checkpoint）step的checkpoint。

[2024/8/18] 预训练已经完成，后续进行微调以及评测

[2024/7/18] 使用8*H800继续训练，wandb：https://api.wandb.ai/links/steel-llm-lab/vqf297nr

[2024/6/30] 放出预训练200k个step的checkpoint，[huggingface链接](https://huggingface.co/gqszhanshijin/Steel-LLM/tree/main)

[2024/5/21] 模型开启正式训练，后续不定期放出checkpoint。

[2024/5/19] 基于Qwen1.5完成模型修改，模型大小1.12B：
- FFN层使用softmax moe，相同参数量下有更高的训练速度
- 使用双层的SwiGLU

相关博客:https://zhuanlan.zhihu.com/p/700395878

[2024/5/5] 预训练程序修改相关的博客：https://zhuanlan.zhihu.com/p/694223107

[2024/4/24] 完成训练程序改进：兼容Hugginface格式模型、支持数据断点续训、支持追加新的数据 

[2024/4/14] 完成数据收集与处理，生成预训练程序所需要的bin文件。更新数据收集与处理相关的博客：https://zhuanlan.zhihu.com/p/687338497


### 🧑‍🤝‍🧑 交流
欢迎加入交流群,人数已超过200，添加微信入群：a1843450905

<br>

## 🤖 预训练
### 数据收集
使用的数据集和链接如下所示，更详细的介绍见[**此篇文章**](https://zhuanlan.zhihu.com/p/687338497)

- [Skywork/Skypile-150B数据集](https://huggingface.co/datasets/Skywork/SkyPile-150B/tree/main/data)
- [wanjuan1.0(nlp部分)](https://opendatalab.org.cn/OpenDataLab/WanJuan1_dot_0?source=Q1NETg)
- [中文维基过滤数据](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- [百度百科数据](https://huggingface.co/datasets/xuqinyang/BaiduBaike-5.63M)
- [百度百科问答数据](https://aistudio.baidu.com/datasetdetail/107726)
- [知乎问答数据](https://huggingface.co/datasets/wangrui6/Zhihu-KOL)
- [BELLE对话数据](https://github.com/LianjiaTech/BELLE/tree/main/data/10M)
- [moss项目对话数据](https://hf-mirror.com/datasets/YeungNLP/moss-003-sft-data)
- [firefly1.1M](https://hf-mirror.com/datasets/YeungNLP/firefly-train-1.1M)
- [starcoder](https://hf-mirror.com/datasets/bigcode/starcoderdata)

### 数据处理
(详细内容见[**此篇文章**](https://mp.weixin.qq.com/s/yqmtHLuuNV9075qHgzhcPw))
#### step1：格式转化
- 源数据：针对4类数据进行格式统一的转化处理：
  - 简单文本：百度百科（title和各段落需要手动合并）、中文维基
  - 对话（含单轮与多轮）：百度百科问答数据、BELLE对话数据（BELLE_3_5M）、moss项目对话数据、知乎问答数据、BELLE任务数据（BELLE_2_5M)、firefly1.1M
  - 代码数据：starcode
  - 其他数据：wanjuan和skypile数据集不用做单独处理
- 目标格式：`{"text": "asdfasdf..."}`，文件保存为.jsonl类型。
- 运行方式：`python data/pretrain_data_prepare/step1_data_process.py`
#### step2：data-juicer数据处理
- 运行方式：`sh data/pretrain_data_prepare/step2/run_step2.sh`

- 具体使用的data juicer算子见<a href=".github/datajuicer_op.md">此文档</a>。

#### step3：生成最终用于训练的bin格式
需要先在代码中修改filename_sets，指定数据路径，然后运行如下程序：

`python pretrain_modify_from_TinyLlama/scripts/prepare_steel_llm_data.py`

输入数据格式为：包含'text'字段的jsonl文件

### tokenizer
不单独训练tokenizer，使用[Qwen/Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)的tokenizer

### 模型结构
(详细内容见[**此篇文章**](https://mp.weixin.qq.com/s/JaZyf1jOEOtNDCcFqSj8TQ))

基于Qwen1.5模型，进行了如下改动：
- FFN层使用softmax moe，相同参数量下有更高的训练速度
- 使用双层的SwiGLU

### 预训练框架
(详细内容见[**此篇文章**](https://mp.weixin.qq.com/s/KPRir6bK3MZZ-vMFTfhUQQ))

基于TinyLlama预训练程序进行如下改进：

  - 兼容HuggingFace格式的模型
  - 加载checkpoint时，完全恢复数据训练的进度
  - 数据一致性检测
  - 在不影响已训练数据的情况下，在数据集中追加新的数据

启动预训练：

`python Steel-LLM/pretrain_modify_from_TinyLlama/pretrain/pretrain_steel_llm.py`

### 评估
(详细内容见[**此篇文章**](https://mp.weixin.qq.com/s/KK0G0spNw0D9rPUESkHMew))

Steel-LLM在CEVAL、CMMLU上进行了测试。Steel-LLM旨在训练一个中文LLM，80%的训练数据都是中文，因此在英文benchmark并未做过多的测试。
其他模型的指标来自于CEVAL论文、MiniCPM技术报告、MAP-Neo技术报告等途径。更多模型的指标可查看之前的<a href=https://mp.weixin.qq.com/s/KK0G0spNw0D9rPUESkHMew>博客</a>

|                              | CEVAL  | CMMLU |
|------------------------------|--------|-------|
| Steel-LLM-chat-v2            | 41.90  | 36.08 |
| Steel-LLM-chat-v1            | 38.57  | 33.48 |
| Tiny-Llama-1.1B              | 25.02  | 24.03 |
| Gemma-2b-it                  | 32.3   | 33.07 |
| Phi2(2B)	                   | 23.37	| 24.18 |
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

## ⛏️ 快速使用
```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "zhanshijin/Steel-LLM"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "你是谁开发的"
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

### 硬件资源
GPU：8* H800 80G(训练30天左右)

GPU：8* A100 80G（训练60天左右）

硬盘：4TB

## Citation

**BibTeX:**
```bibtex
@article{gu2025steel,
  title={Steel-LLM: From Scratch to Open Source--A Personal Journey in Building a Chinese-Centric LLM},
  author={Gu, Qingshui and Li, Shu and Zheng, Tianyu and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2502.06635},
  year={2025}
}
```