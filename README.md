<div align="center">

# 开源中文预训练语言模型Steel-LLM
由zhanshijin和lishu14创建
</div>


## 👋 介绍
Steel-LLM是一个从0开始预训练中文大模型的项目。我们的目标是使用1T+的数据预训练一个1B左右参数量的中文LLM，对标TinyLlama。我们会分享数据收集、数据处理、预训练框架选择、模型设计等全过程，并开源全部代码。让每个人在有8~几十张卡的情况下都能复现我们的工作。"Steel(钢)"取名灵感来源于国内一只优秀的乐队“万能青年旅店（万青）”。乐队在做一专的时候条件有限，自称为是在“土法炼钢”，但却是一张神专。我们训练LLM的条件同样有限，但也希望能炼出好“钢”来。为了让能持续关注我们的同志们有一些参与感，并在未来使用Steel-LLM时让模型更加懂你，我们会持续收集大家的数据，各种亚文化、冷知识、歌词、小众读物、只有你自己知道的小秘密等等都可以，并训练到我们的LLM中。改编万青一专介绍结束语作为我们介绍的结束语：Steel-LLM完成之时，神经元已经被万亿数据填满。我们渴望这个塞了很多东西的模型还能为你们的数据留下丝缕空地。这样的话，所有用到模型的人，就有可能并肩站在一起。

## 🔔 公告 
### 数据收集
可以将想训练进模型的数据在问卷中填写。文本不长可以直接粘贴进问卷，如果文本较长最好能够存在txt、json等可直接能被程序读取的文件中并上传。PDF等不好处理的文件将会在项目后期进行处理并训练。问卷链接：（上传的数据内容请遵守各项法规）

### 更新

### 技术分享
zhanshijin的知乎：https://www.zhihu.com/people/zhan-shi-jin-27

lishu14的知乎：
https://www.zhihu.com/people/a-xun-58-5


## 🤖 预训练
### 数据收集
使用的数据集和链接如下所示，更详细的介绍请看**此篇文章**

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


### tokenizer
不单独训练tokenizer，初步决定使用Qwen的tokenizer

### 模型结构
👷 待定，施工中...

### 预训框架
👷 施工中... 主要基于TinyLlama的训练程序并进行修改，目前已完成通过多进程将原始文本token化并存储到bin文件中的工作。

### 硬件资源
GPU：8* A100 80G
硬盘：4TB

Train LLM from 0 By personal.Update from 2024.3.

## 交流