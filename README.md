<div align="center">

# 开源中文预训练语言模型Steel-LLM
由zhanshijin和lishu14创建
</div>


## 👋 介绍
Steel-LLM是一个从零开始预训练中文大模型的项目。我们的目标是使用1T+的数据预训练一个1B左右参数量的中文LLM，对标TinyLlama。项目持续更新，维持3个月+。我们会分享数据收集、数据处理、预训练框架选择、模型设计等全过程，并开源全部代码。让每个人在有8~几十张卡的情况下都能复现我们的工作。
<div align="center">
  <img src=".github/steel.png" width="200"/>
</div>
<p align="center">
        🤗 <a href="https://huggingface.co/gqszhanshijin/Steel-LLM">Hugging Face</a>&nbsp&nbsp 
        &nbsp&nbsp 📑 <a href="https://www.zhihu.com/people/zhan-shi-jin-27">Blog</a>

"Steel(钢)"取名灵感来源于华北平原一只优秀的乐队“万能青年旅店（万青）”。乐队在做一专的时候条件有限，自称是在“土法炼钢”，但却是一张神专。我们训练LLM的条件同样有限，但也希望能炼出好“钢”来。

## 🔔 公告 

### 更新
[2024/9/2] HuggingFace更新了480k、660k、720k、980k、1060k（最后一个checkpoint）step的checkpoint。

[2024/8/18] 预训练已经完成，后续进行微调以及评测

[2024/7/18] 使用8*H800继续训练，wandb：https://wandb.ai/steel-llm-lab/lightning_logs/reports/Untitled-Report--Vmlldzo4NzI1MTQz

[2024/6/30] 放出预训练200k个step的checkpoint，[huggingface链接](https://huggingface.co/gqszhanshijin/Steel-LLM/tree/main)

[2024/5/21] 模型开启正式训练，后续不定期放出checkpoint。

[2024/5/19] 基于Qwen1.5完成模型修改，模型大小1.12B：
- FFN层使用softmax moe，相同参数量下有更高的训练速度
- 使用双层的SwiGLU

相关博客:https://zhuanlan.zhihu.com/p/700395878

[2024/5/5] 预训练程序修改相关的博客：https://zhuanlan.zhihu.com/p/694223107

[2024/4/24] 完成训练程序改进：兼容Hugginface格式模型、支持数据断点续训、支持追加新的数据 

[2024/4/14] 完成数据收集与处理，生成预训练程序所需要的bin文件。更新数据收集与处理相关的博客：https://zhuanlan.zhihu.com/p/687338497

### 技术分享
zhanshijin的知乎：https://www.zhihu.com/people/zhan-shi-jin-27

lishu14的知乎：https://www.zhihu.com/people/a-xun-58-5


## 🤖 预训练
### 数据收集
使用的数据集和链接如下所示，更详细的介绍请看[**此篇文章**](https://zhuanlan.zhihu.com/p/687338497)

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
#### 格式转化（step1）
- 源数据：针对三类数据进行格式统一的转化处理。
  - 简单文本：百度百科（title和各段落需要手动合并）、中文维基
  - 对话（含单轮与多轮）：百度百科问答数据、BELLE对话数据（BELLE_3_5M）、moss项目对话数据、知乎问答数据
  - 任务：BELLE任务数据（BELLE_2_5M)、firefly1.1M
  - 代码数据：starcode
- 目标格式：`{"text": "asdfasdf..."}`，文件保存为.jsonl类型。
- 运行方式：`python data/pretrain_data_prepare/step1_data_process.py`
#### data-juicer数据处理（step2）
我们使用data-juicer处理文本时，不改变数据格式。
- 运行方式：`sh data/pretrain_data_prepare/step2/run_step2.sh`

- 选用的文本处理算子


|算子|描述|
|:----|:----|
|chinese_convert_mapper|用于在繁体中文、简体中文和日文汉字之间进行转换（借助 opencc）|
|clean_email_mapper|删除邮箱信息|
|clean_html_mapper|删除 HTML 标签并返回所有节点的纯文本|
|clean_ip_mapper|删除 IP 地址|
|clean_links_mapper|删除链接，例如以 http 或 ftp 开头的|
|clean_copyright_mapper|删除代码文件开头的版权声明 (:warning: 必须包含单词 copyright)|
|expand_macro_mapper|扩展通常在 TeX 文档顶部定义的宏|
|fix_unicode_mapper|修复损坏的 Unicode（借助 ftfy）|
|punctuation_normalization_mapper|将各种 Unicode 标点符号标准化为其 ASCII 等效项|
|remove_repeat_sentences_mapper|删除样本中的重复句子|
|remove_specific_chars_mapper|删除样本中的特殊字符（用户自定义）|
|whitespace_normalization_mapper|将各类空格归一转换为英语空格|
|alphanumeric_filter|保留字母数字比例在指定范围内的样本|
|average_line_length_filter|保留平均行长度在指定范围内的样本|
|character_repetition_filter|保留 char-level n-gram 重复比率在指定范围内的样本|
|maximum_line_length_filter|保留最大行长度在指定范围内的样本|
|perplexity_filter|保留困惑度低于指定阈值的样本|
|special_characters_filter|保留 special-char 比率的在指定范围内的样本|
|text_length_filter|保留总文本长度在指定范围内的样本|
|word_repetition_filter|保留 word-level n-gram 重复比率在指定范围内的样本|
|document_simhash_deduplicator|使用 SimHash 在文档级别对样本去重|


- 选用的代码处理算子
  

|算子|描述|
|:----|:----|
|clean_copyright_mapper|删除代码文件开头的版权声明 (:warning: 必须包含单词 copyright)|
|clean_email_mapper|删除邮箱信息|
|clean_links_mapper|删除链接，例如以 http 或 ftp 开头的|
|fix_unicode_mapper|修复损坏的 Unicode（借助 ftfy）|
|punctuation_normalization_mapper|将各种 Unicode 标点符号标准化为其 ASCII 等效项|
|alphanumeric_filter|保留字母数字比例在指定范围内的样本|
|average_line_length_filter|保留平均行长度在指定范围内的样本|
|character_repetition_filter|保留 char-level n-gram 重复比率在指定范围内的样本|
|maximum_line_length_filter|保留最大行长度在指定范围内的样本|
|text_length_filter|保留总文本长度在指定范围内的样本|
|word_num_filter|保留字数在指定范围内的样本|
|word_repetition_filter|保留 word-level n-gram 重复比率在指定范围内的样本|
|document_simhash_deduplicator|使用 SimHash 在文档级别对样本去重|

#### 生成最终用于训练的bin格式
需要先在代码中修改filename_sets，指定数据路径：

`python pretrain_modify_from_TinyLlama/scripts/prepare_steel_llm_data.py`

输入数据格式为：包含'text'字段的jsonl文件

### tokenizer
不单独训练tokenizer，使用[Qwen/Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)的tokenizer

### 模型结构
基于Qwen1.5模型，进行了如下改动：
- FFN层使用softmax moe，相同参数量下有更高的训练速度
- 使用双层的SwiGLU

### 预训框架

基于TinyLlama预训练程序进行如下改进：

  - 兼容HuggingFace格式的模型
  - 加载checkpoint时，完全恢复数据训练的进度
  - 数据一致性检测
  - 在不影响已训练数据的情况下，在数据集中追加新的数据

启动预训练：

`python Steel-LLM/pretrain_modify_from_TinyLlama/pretrain/pretrain_steel_llm.py`


### 硬件资源
GPU：8* H800 80G

~~GPU：8* A100 80G~~
硬盘：4TB


## 🧑‍🤝‍🧑 交流
欢迎加入交流群,人数已超过200，添加微信入群：a1843450905


