---
language:
- zh
- en
tags:
- qwen
pipeline_tag: text-generation
inference: false
---

# Qwen-1.8B-Chat

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/logo_qwen.jpg" width="400"/>
<p>
<br>

<p align="center">
        🤗 <a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2309.16609">Paper</a> &nbsp&nbsp ｜ &nbsp&nbsp🖥️ <a href="https://www.modelscope.cn/studios/qwen/Qwen-1_8B-Chat-Demo/summary">Demo</a>
<br>
<a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://discord.gg/z3GAxXZ9Ce">Discord</a>&nbsp&nbsp ｜  &nbsp&nbsp<a href="https://dashscope.aliyun.com">API</a> 
</p>
<br>

## 介绍（Introduction）
**通义千问-1.8B（Qwen-1.8B）**是阿里云研发的通义千问大模型系列的18亿参数规模的模型。Qwen-1.8B是基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。同时，在Qwen-1.8B的基础上，我们使用对齐机制打造了基于大语言模型的AI助手Qwen-1.8B-Chat。本仓库为Qwen-1.8B-Chat的仓库。

通义千问-1.8B（Qwen-1.8B）主要有以下特点：
1. **低成本部署**：提供int8和int4量化版本，推理最低仅需不到2GB显存，生成2048 tokens仅需3GB显存占用。微调最低仅需6GB。
2. **大规模高质量训练语料**：使用超过2.2万亿tokens的数据进行预训练，包含高质量中、英、多语言、代码、数学等数据，涵盖通用及专业领域的训练语料。通过大量对比实验对预训练语料分布进行了优化。
3. **优秀的性能**：Qwen-1.8B支持8192上下文长度，在多个中英文下游评测任务上（涵盖常识推理、代码、数学、翻译等），效果显著超越现有的相近规模开源模型，具体评测结果请详见下文。
4. **覆盖更全面的词表**：相比目前以中英词表为主的开源模型，Qwen-1.8B使用了约15万大小的词表。该词表对多语言更加友好，方便用户在不扩展词表的情况下对部分语种进行能力增强和扩展。
5. **系统指令跟随**：Qwen-1.8B-Chat可以通过调整系统指令，实现**角色扮演**，**语言风格迁移**，**任务设定**，和**行为设定**等能力。


如果您想了解更多关于通义千问1.8B开源模型的细节，我们建议您参阅[GitHub代码库](https://github.com/QwenLM/Qwen)。

**Qwen-1.8B** is the 1.8B-parameter version of the large language model series, Qwen (abbr. Tongyi Qianwen), proposed by Aibaba Cloud. Qwen-1.8B is a Transformer-based large language model, which is pretrained on a large volume of data, including web texts, books, codes, etc. Additionally, based on the pretrained Qwen-1.8B, we release Qwen-1.8B-Chat, a large-model-based AI assistant, which is trained with alignment techniques. This repository is the one for Qwen-1.8B-Chat.

The features of Qwen-1.8B include:
1. **Low-cost deployment**: We provide int4 and int8 quantized versions, the minimum memory requirment for inference is less than 2GB, generating 2048 tokens only 3GB of memory usage. The minimum memory requirment of finetuning is only 6GB.

2. **Large-scale high-quality training corpora**: It is pretrained on over 2.2 trillion tokens, including Chinese, English, multilingual texts, code, and mathematics, covering general and professional fields. The distribution of the pre-training corpus has been optimized through a large number of ablation experiments.
3. **Good performance**: It supports 8192 context length and significantly surpasses existing open-source models of similar scale on multiple Chinese and English downstream evaluation tasks (including commonsense, reasoning, code, mathematics, etc.), and even surpasses some larger-scale models in several benchmarks. See below for specific evaluation results.
4. **More comprehensive vocabulary coverage**: Compared with other open-source models based on Chinese and English vocabularies, Qwen-1.8B uses a vocabulary of over 150K tokens. This vocabulary is more friendly to multiple languages, enabling users to directly further enhance the capability for certain languages without expanding the vocabulary.
5. **System prompt**: Qwen-1.8B-Chat can realize roly playing, language style transfer, task setting, and behavior setting by using system prompt.

For more details about the open-source model of Qwen-1.8B-Chat, please refer to the [GitHub](https://github.com/QwenLM/Qwen) code repository.
                                                                                        

<br>

## 要求（Requirements）

* python 3.8及以上版本
* pytorch 1.12及以上版本，推荐2.0及以上版本
* 建议使用CUDA 11.4及以上（GPU用户、flash-attention用户等需考虑此选项）
* python 3.8 and above
* pytorch 1.12 and above, 2.0 and above are recommended
* CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)

## 依赖项（Dependency）

运行Qwen-1.8B-Chat，请确保满足上述要求，再执行以下pip命令安装依赖库

To run Qwen-1.8B-Chat, please make sure you meet the above requirements, and then execute the following pip commands to install the dependent libraries.

```bash
pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
```

另外，推荐安装`flash-attention`库（**当前已支持flash attention 2**），以实现更高的效率和更低的显存占用。

In addition, it is recommended to install the `flash-attention` library (**we support flash attention 2 now.**) for higher efficiency and lower memory usage.

```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# 下方安装可选，安装可能比较缓慢。
# pip install csrc/layer_norm
# pip install csrc/rotary
```
<br>

## 快速使用（Quickstart）

下面我们展示了一个使用Qwen-1.8B-Chat模型，进行多轮对话交互的样例：

We show an example of multi-turn interaction with Qwen-1.8B-Chat in the following code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="cpu", trust_remote_code=True).eval()
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="auto", trust_remote_code=True).eval()

# Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

# 第一轮对话 1st dialogue turn
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。

# 第二轮对话 2nd dialogue turn
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
# 这是一个关于一个年轻人奋斗创业最终取得成功的故事。
# 故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。从小，李明就立下了一个目标：要成为一名成功的企业家。
# 为了实现这个目标，李明勤奋学习，考上了大学。在大学期间，他积极参加各种创业比赛，获得了不少奖项。他还利用课余时间去实习，积累了宝贵的经验。
# 毕业后，李明决定开始自己的创业之路。他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。他继续努力，不断改进自己的创业计划，并寻找新的投资机会。
# 最终，李明成功地获得了一笔投资，开始了自己的创业之路。他成立了一家科技公司，专注于开发新型软件。在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。
# 李明的成功并不是偶然的。他勤奋、坚韧、勇于冒险，不断学习和改进自己。他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。

# 第三轮对话 3rd dialogue turn
response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
print(response)
# 《奋斗创业：一个年轻人的成功之路》

# Qwen-1.8B-Chat现在可以通过调整系统指令（System Prompt），实现角色扮演，语言风格迁移，任务设定，行为设定等能力。
# Qwen-1.8B-Chat can realize roly playing, language style transfer, task setting, and behavior setting by system prompt.
response, _ = model.chat(tokenizer, "你好呀", history=None, system="请用二次元可爱语气和我说话")
print(response)
# 你好啊！我是一只可爱的二次元猫咪哦，不知道你有什么问题需要我帮忙解答吗？

response, _ = model.chat(tokenizer, "My colleague works diligently", history=None, system="You will write beautiful compliments according to needs")
print(response)
# Your colleague is an outstanding worker! Their dedication and hard work are truly inspiring. They always go above and beyond to ensure that 
# their tasks are completed on time and to the highest standard. I am lucky to have them as a colleague, and I know I can count on them to handle any challenge that comes their way.
```

关于更多的使用说明，请参考我们的[GitHub repo](https://github.com/QwenLM/Qwen)获取更多信息。

For more information, please refer to our [GitHub repo](https://github.com/QwenLM/Qwen) for more information.

## Tokenizer

> 注：作为术语的“tokenization”在中文中尚无共识的概念对应，本文档采用英文表达以利说明。

基于tiktoken的分词器有别于其他分词器，比如sentencepiece分词器。尤其在微调阶段，需要特别注意特殊token的使用。关于tokenizer的更多信息，以及微调时涉及的相关使用，请参阅[文档](https://github.com/QwenLM/Qwen/blob/main/tokenization_note_zh.md)。

Our tokenizer based on tiktoken is different from other tokenizers, e.g., sentencepiece tokenizer. You need to pay attention to special tokens, especially in finetuning. For more detailed information on the tokenizer and related use in fine-tuning, please refer to the [documentation](https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md).

## 量化 (Quantization)

### 用法 (Usage)

**请注意：我们更新量化方案为基于[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)的量化，提供Qwen-1.8B-Chat的Int4量化模型[点击这里](https://huggingface.co/Qwen/Qwen-1_8B-Chat-Int4)。相比此前方案，该方案在模型评测效果几乎无损，且存储需求更低，推理速度更优。**

**Note: we provide a new solution based on [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), and release an Int4 quantized model for Qwen-1.8B-Chat [Click here](https://huggingface.co/Qwen/Qwen-1_8B-Chat-Int4), which achieves nearly lossless model effects but improved performance on both memory costs and inference speed, in comparison with the previous solution.**

以下我们提供示例说明如何使用Int4量化模型。在开始使用前，请先保证满足要求（如torch 2.0及以上，transformers版本为4.32.0及以上，等等），并安装所需安装包：

Here we demonstrate how to use our provided quantized models for inference. Before you start, make sure you meet the requirements of auto-gptq (e.g., torch 2.0 and above, transformers 4.32.0 and above, etc.) and install the required packages:

```bash
pip install auto-gptq optimum
```

如安装`auto-gptq`遇到问题，我们建议您到官方[repo](https://github.com/PanQiWei/AutoGPTQ)搜索合适的预编译wheel。

随后即可使用和上述一致的用法调用量化模型：

If you meet problems installing `auto-gptq`, we advise you to check out the official [repo](https://github.com/PanQiWei/AutoGPTQ) to find a pre-build wheel.

Then you can load the quantized model easily and run inference as same as usual:

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-1_8B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "你好", history=None)
```

### 效果评测

我们使用原始模型的FP32和BF16精度，以及量化过的Int8和Int4模型在基准评测上做了测试，结果如下所示：

We illustrate the model performance of both FP32, BF16, Int8 and Int4 models on the benchmark. Results are shown below:

| Quantization | MMLU | CEval (val) | GSM8K | Humaneval |
|--------------|:----:|:-----------:|:-----:|:---------:|
| FP32         | 43.4 |    57.0     | 33.0  |   26.8    |
| BF16         | 43.3 |    55.6     | 33.7  |   26.2    |
| Int8         | 43.1 |    55.8     | 33.0  |   27.4    |
| Int4         | 42.9 |    52.8     | 31.2  |   25.0    |

### 推理速度 (Inference Speed)

我们测算了FP32、BF16精度和Int8、Int4量化模型生成2048和8192个token的平均推理速度。如图所示：

We measured the average inference speed of generating 2048 and 8192 tokens under FP32, BF16 precision and Int8, Int4 quantization level, respectively.

| Quantization | FlashAttn | Speed (2048 tokens) | Speed (8192 tokens) |
|--------------| :-------: |:-------------------:|:-------------------:|
| FP32         |   v2      |        52.96        |        47.35        |
| BF16         |   v2      |        54.09        |        54.04        |
| Int8         |   v2      |        55.56        |        55.62        |
| Int4         |   v2      |        71.07        |        76.45        |
| FP32         |   v1      |        52.00        |        45.80        |
| BF16         |   v1      |        51.70        |        55.04        |
| Int8         |   v1      |        53.16        |        53.33        |
| Int4         |   v1      |        69.82        |        67.44        |
| FP32         |  Disabled |        52.28        |        44.95        |
| BF16         |  Disabled |        48.17        |        45.01        |
| Int8         |  Disabled |        52.16        |        52.99        |
| Int4         |  Disabled |        68.37        |        65.94        |

具体而言，我们记录在长度为1的上下文的条件下生成8192个token的性能。评测运行于单张A100-SXM4-80G GPU，使用PyTorch 2.0.1和CUDA 11.4。推理速度是生成8192个token的速度均值。

In detail, the setting of profiling is generating 8192 new tokens with 1 context token. The profiling runs on a single A100-SXM4-80G GPU with PyTorch 2.0.1 and CUDA 11.4. The inference speed is averaged over the generated 8192 tokens.

### 显存使用 (GPU Memory Usage)

我们测算了FP32、BF16精度和Int8、Int4量化模型生成2048个及8192个token（单个token作为输入）的峰值显存占用情况。结果如下所示：

We also profile the peak GPU memory usage for generating 2048 tokens and 8192 tokens (with single token as context) under FP32, BF16 or Int8, Int4 quantization level, respectively. The results are shown below.

| Quantization Level | Peak Usage for Encoding 2048 Tokens | Peak Usage for Generating 8192 Tokens |
|--------------------|:-----------------------------------:|:-------------------------------------:|
| FP32               |               8.45GB                |                13.06GB                |
| BF16               |               4.23GB                |                6.48GB                 |
| Int8               |               3.48GB                |                5.34GB                 |
| Int4               |               2.91GB                |                4.80GB                 |

上述性能测算使用[此脚本](https://qianwen-res.oss-cn-beijing.aliyuncs.com/profile.py)完成。

The above speed and memory profiling are conducted using [this script](https://qianwen-res.oss-cn-beijing.aliyuncs.com/profile.py).
<br>

## 模型细节（Model）

与Qwen-1.8B预训练模型相同，Qwen-1.8B-Chat模型规模基本情况如下所示

The details of the model architecture of Qwen-1.8B-Chat are listed as follows

| Hyperparameter  | Value  |
|:----------------|:------:|
| n_layers        |   24   |
| n_heads         |   16   |
| d_model         |  2048  |
| vocab size      | 151851 |
| sequence length |  8192  |

在位置编码、FFN激活函数和normalization的实现方式上，我们也采用了目前最流行的做法，
即RoPE相对位置编码、SwiGLU激活函数、RMSNorm（可选安装flash-attention加速）。

在分词器方面，相比目前主流开源模型以中英词表为主，Qwen-1.8B-Chat使用了约15万token大小的词表。
该词表在GPT-4使用的BPE词表`cl100k_base`基础上，对中文、多语言进行了优化，在对中、英、代码数据的高效编解码的基础上，对部分多语言更加友好，方便用户在不扩展词表的情况下对部分语种进行能力增强。
词表对数字按单个数字位切分。调用较为高效的[tiktoken分词库](https://github.com/openai/tiktoken)进行分词。

For position encoding, FFN activation function, and normalization calculation methods, we adopt the prevalent practices, i.e., RoPE relative position encoding, SwiGLU for activation function, and RMSNorm for normalization (optional installation of flash-attention for acceleration).

For tokenization, compared to the current mainstream open-source models based on Chinese and English vocabularies, Qwen-1.8B-Chat uses a vocabulary of over 150K tokens.
It first considers efficient encoding of Chinese, English, and code data, and is also more friendly to multilingual languages, enabling users to directly enhance the capability of some languages without expanding the vocabulary.
It segments numbers by single digit, and calls the [tiktoken](https://github.com/openai/tiktoken) tokenizer library for efficient tokenization.

## 评测效果（Evaluation）

对于Qwen-1.8B-Chat模型，我们同样评测了常规的中文理解（C-Eval）、英文理解（MMLU）、代码（HumanEval）和数学（GSM8K）等权威任务，同时包含了长序列任务的评测结果。由于Qwen-1.8B-Chat模型经过对齐后，激发了较强的外部系统调用能力，我们还进行了工具使用能力方面的评测。

提示：由于硬件和框架造成的舍入误差，复现结果如有波动属于正常现象。

For Qwen-1.8B-Chat, we also evaluate the model on C-Eval, MMLU, HumanEval, GSM8K, etc., as well as the benchmark evaluation for long-context understanding, and tool usage.

Note: Due to rounding errors caused by hardware and framework, differences in reproduced results are possible.

### 中文评测（Chinese Evaluation）

#### C-Eval

在[C-Eval](https://arxiv.org/abs/2305.08322)验证集上，我们评价了Qwen-1.8B-Chat模型的准确率

We demonstrate the accuracy of Qwen-1.8B-Chat on C-Eval validation set

|          Model                   |    Acc.   |
|:--------------------------------:|:---------:|
| RedPajama-INCITE-Chat-3B         |   18.3    |
|       OpenBuddy-3B               |   23.5    |
|    Firefly-Bloom-1B4             |   23.6    |
|   OpenLLaMA-Chinese-3B           |   24.4    |
|          LLaMA2-7B-Chat          |   31.9    |
|         ChatGLM2-6B-Chat         |   52.6    |
|         InternLM-7B-Chat         |   53.6    |
|    **Qwen-1.8B-Chat (0-shot)**   |   55.6    |
|    **Qwen-7B-Chat (0-shot)**     |   59.7    |
|    **Qwen-7B-Chat (5-shot)**     |   59.3    |

C-Eval测试集上，Qwen-1.8B-Chat模型的zero-shot准确率结果如下：

The zero-shot accuracy of Qwen-1.8B-Chat on C-Eval testing set is provided below:

| Model                   |   Avg.   | STEM | Social Sciences | Humanities | Others |
| :---------------------: | :------: | :--: | :-------------: | :--------: | :----: |
| Chinese-Alpaca-Plus-13B |   41.5   | 36.6 |      49.7       |    43.1    |  41.2  |
| Chinese-Alpaca-2-7B     |   40.3   |  -   |        -        |     -      |   -    |
| ChatGLM2-6B-Chat        |   50.1   | 46.4 |      60.4       |    50.6    |  46.9  |
| Baichuan-13B-Chat       |   51.5   | 43.7 |      64.6       |    56.2    |  49.2  |
| **Qwen-1.8B-Chat**      |   53.8   | 48.4 |      68.0       |    56.5    |  48.3  |
| **Qwen-7B-Chat**        |   58.6   | 53.3 |      72.1       |    62.8    |  52.0  |

### 英文评测（English Evaluation）

#### MMLU

[MMLU](https://arxiv.org/abs/2009.03300)评测集上，Qwen-1.8B-Chat模型的准确率如下，效果同样在同类对齐模型中同样表现较优。

The accuracy of Qwen-1.8B-Chat on MMLU is provided below.
The performance of Qwen-1.8B-Chat still on the top between other human-aligned models with comparable size.

|          Model                   |   Acc.    |
|:--------------------------------:|:---------:|
|    Firefly-Bloom-1B4             |   23.8    |
|       OpenBuddy-3B               |   25.5    |
| RedPajama-INCITE-Chat-3B         |   25.5    |
|   OpenLLaMA-Chinese-3B           |   25.7    |
|         ChatGLM2-6B-Chat         |   46.0    |
|          LLaMA2-7B-Chat          |   46.2    |
|         InternLM-7B-Chat         |   51.1    |
|        Baichuan2-7B-Chat         |   52.9    |
|    **Qwen-1.8B-Chat (0-shot)**   |   43.3    |
|    **Qwen-7B-Chat (0-shot)**     |   55.8    |
|    **Qwen-7B-Chat (5-shot)**     |   57.0    |

### 代码评测（Coding Evaluation）

Qwen-1.8B-Chat在[HumanEval](https://github.com/openai/human-eval)的zero-shot Pass@1效果如下

The zero-shot Pass@1 of Qwen-1.8B-Chat on [HumanEval](https://github.com/openai/human-eval) is demonstrated below

|          Model           | Pass@1 |
|:------------------------:|:------:|
|    Firefly-Bloom-1B4     |  0.6   |
|   OpenLLaMA-Chinese-3B   |  4.9   |
| RedPajama-INCITE-Chat-3B |  6.1   |
|       OpenBuddy-3B       |  10.4  |
|    ChatGLM2-6B-Chat      |  11.0  |
|     LLaMA2-7B-Chat       |  12.2  |
|    Baichuan2-7B-Chat     |  13.4  |
|    InternLM-7B-Chat      |  14.6  |
|    **Qwen-1.8B-Chat**    |  26.2  |
|    **Qwen-7B-Chat**      |  37.2  |

### 数学评测（Mathematics Evaluation）

在评测数学能力的[GSM8K](https://github.com/openai/grade-school-math)上，Qwen-1.8B-Chat的准确率结果如下

The accuracy of Qwen-1.8B-Chat on GSM8K is shown below

|                 Model                |    Acc.  |
|:------------------------------------:|:--------:|
|         Firefly-Bloom-1B4            |   2.4    |
|      RedPajama-INCITE-Chat-3B        |   2.5    |
|         OpenLLaMA-Chinese-3B         |   3.0    |
|            OpenBuddy-3B              |   12.6   |
|            LLaMA2-7B-Chat            |   26.3   |
|           ChatGLM2-6B-Chat           |   28.8   |
|          Baichuan2-7B-Chat           |   32.8   |
|           InternLM-7B-Chat           |   33.0   |
|    **Qwen-1.8B-Chat (0-shot)**       |   33.7   |
|      **Qwen-7B-Chat (0-shot)**       |   50.3   |
|      **Qwen-7B-Chat (8-shot)**       |   54.1   |

## 评测复现（Reproduction）

我们提供了评测脚本，方便大家复现模型效果，详见[链接](https://github.com/QwenLM/Qwen/tree/main/eval)。提示：由于硬件和框架造成的舍入误差，复现结果如有小幅波动属于正常现象。

We have provided evaluation scripts to reproduce the performance of our model, details as [link](https://github.com/QwenLM/Qwen/tree/main/eval).
<br>

## FAQ

如遇到问题，敬请查阅[FAQ](https://github.com/QwenLM/Qwen/blob/main/FAQ_zh.md)以及issue区，如仍无法解决再提交issue。

If you meet problems, please refer to [FAQ](https://github.com/QwenLM/Qwen/blob/main/FAQ.md) and the issues first to search a solution before you launch a new issue.
<br>

## 引用 (Citation)

如果你觉得我们的工作对你有帮助，欢迎引用！

If you find our work helpful, feel free to give us a cite.

```
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```
<br>

## 使用协议（License Agreement）

我们的代码和模型权重对学术研究完全开放。请查看[LICENSE](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20RESEARCH%20LICENSE%20AGREEMENT)文件了解具体的开源协议细节。如需商用，请联系我们。

Our code and checkpoints are open to research purpose. Check the [LICENSE](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20RESEARCH%20LICENSE%20AGREEMENT) for more details about the license. For commercial use, please contact us.
<br>

## 联系我们（Contact Us）

如果你想给我们的研发团队和产品团队留言，欢迎加入我们的微信群、钉钉群以及Discord！同时，也欢迎通过邮件（qianwen_opensource@alibabacloud.com）联系我们。

If you are interested to leave a message to either our research team or product team, join our Discord or WeChat groups! Also, feel free to send an email to qianwen_opensource@alibabacloud.com.

