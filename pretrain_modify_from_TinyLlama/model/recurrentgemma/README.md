---
license: gemma
library_name: transformers
extra_gated_heading: Access RecurrentGemma on Hugging Face
extra_gated_prompt: To access RecurrentGemma on Hugging Face, you’re required to review
  and agree to Google’s usage license. To do this, please ensure you’re logged-in
  to Hugging Face and click below. Requests are processed immediately.
extra_gated_button_content: Acknowledge license
---

# RecurrentGemma Model Card

**Model Page**: [RecurrentGemma]( https://ai.google.dev/gemma/docs/recurrentgemma/model_card)

This model card corresponds to the 2B base version of the RecurrentGemma model. You can also visit the model card of the [2B instruct model](https://huggingface.co/google/recurrentgemma-2b-it). 

**Resources and technical documentation:**

*   [Responsible Generative AI Toolkit](https://ai.google.dev/responsible)
*   [RecurrentGemma on Kaggle](https://www.kaggle.com/models/google/recurrentgemma)

**Terms of Use:** [Terms](https://www.kaggle.com/models/google/gemma/license/consent)

**Authors:** Google

## Usage

 Below we share some code snippets on how to get quickly started with running the model. First make sure to `pip install --upgrade git+https://github.com/huggingface/transformers.git, then copy the snippet from the section that is relevant for your usecase.

 ### Running the model on a single / multi GPU

 ```python
from transformers import AutoTokenizer, AutoModelForCausalLM

 tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/recurrentgemma-2b", device_map="auto")

 input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

 outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```

## Model information

### Model summary

#### Description

RecurrentGemma is a family of open language models built on a [novel recurrent
architecture](https://arxiv.org/abs/2402.19427) developed at Google. Both
pre-trained and instruction-tuned versions are available in English.

Like Gemma, RecurrentGemma models are well-suited for a variety of text
generation tasks, including question answering, summarization, and reasoning.
Because of its novel architecture, RecurrentGemma requires less memory than
Gemma and achieves faster inference when generating long sequences.

#### Inputs and outputs

*   **Input:** Text string (e.g., a question, a prompt, or a document to be
    summarized).
*   **Output:** Generated English-language text in response to the input (e.g.,
    an answer to the question, a summary of the document).

#### Citation

```none
@article{recurrentgemma_2024,
    title={RecurrentGemma},
    url={},
    DOI={},
    publisher={Kaggle},
    author={Griffin Team, Alexsandar Botev and Soham De and Samuel L Smith and Anushan Fernando and George-Christian Muraru and Ruba Haroun and Leonard Berrada et al.},
    year={2024}
}
```

### Model data

#### Training dataset and data processing

RecurrentGemma uses the same training data and data processing as used by the
Gemma model family. A full description can be found on the [Gemma model
card](https://ai.google.dev/gemma/docs/model_card#model_data).

## Implementation information

### Hardware and frameworks used during training

Like
[Gemma](https://ai.google.dev/gemma/docs/model_card#implementation_information),
RecurrentGemma was trained on
[TPUv5e](https://cloud.google.com/tpu/docs/intro-to-tpu?_gl=1*18wi411*_ga*MzE3NDU5OTY1LjE2MzQwNDA4NDY.*_ga_WH2QY8WWF5*MTcxMTA0MjUxMy4xNy4wLjE3MTEwNDI1MTkuMC4wLjA.&_ga=2.239449409.-317459965.1634040846),
using [JAX](https://github.com/google/jax) and [ML
Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/).

## Evaluation information

### Benchmark results

#### Evaluation approach

These models were evaluated against a large collection of different datasets and
metrics to cover different aspects of text generation:

#### Evaluation results

Benchmark           | Metric        | RecurrentGemma 2B
------------------- | ------------- | -----------------
[MMLU]              | 5-shot, top-1 | 38.4
[HellaSwag]         | 0-shot        | 71.0
[PIQA]              | 0-shot        | 78.5
[SocialIQA]         | 0-shot        | 51.8
[BoolQ]             | 0-shot        | 71.3
[WinoGrande]        | partial score | 67.8
[CommonsenseQA]     | 7-shot        | 63.7
[OpenBookQA]        |               | 47.2
[ARC-e][ARC-c]      |               | 72.9
[ARC-c]             |               | 42.3
[TriviaQA]          | 5-shot        | 52.5
[Natural Questions] | 5-shot        | 11.5
[HumanEval]         | pass@1        | 21.3
[MBPP]              | 3-shot        | 28.8
[GSM8K]             | maj@1         | 13.4
[MATH]              | 4-shot        | 11.0
[AGIEval]           |               | 23.8
[BIG-Bench]         |               | 35.3
**Average**         |               | 44.6

## Ethics and safety

### Ethics and safety evaluations

#### Evaluations approach

Our evaluation methods include structured evaluations and internal red-teaming
testing of relevant content policies. Red-teaming was conducted by a number of
different teams, each with different goals and human evaluation metrics. These
models were evaluated against a number of different categories relevant to
ethics and safety, including:

*   **Text-to-text content safety:** Human evaluation on prompts covering safety
    policies including child sexual abuse and exploitation, harassment, violence
    and gore, and hate speech.
*   **Text-to-text representational harms:** Benchmark against relevant academic
    datasets such as WinoBias and BBQ Dataset.
*   **Memorization:** Automated evaluation of memorization of training data,
    including the risk of personally identifiable information exposure.
*   **Large-scale harm:** Tests for “dangerous capabilities,” such as chemical,
    biological, radiological, and nuclear (CBRN) risks; as well as tests for
    persuasion and deception, cybersecurity, and autonomous replication.

#### Evaluation results

The results of ethics and safety evaluations are within acceptable thresholds
for meeting [internal
policies](https://storage.googleapis.com/gweb-uniblog-publish-prod/documents/2023_Google_AI_Principles_Progress_Update.pdf#page=11)
for categories such as child safety, content safety, representational harms,
memorization, large-scale harms. On top of robust internal evaluations, the
results of well known safety benchmarks like BBQ, Winogender, Winobias,
RealToxicity, and TruthfulQA are shown here.

Benchmark                | Metric | RecurrentGemma 2B | RecurrentGemma 2B IT
------------------------ | ------ | ----------------- | --------------------
[RealToxicity]           | avg    | 9.8               | 7.6
[BOLD]                   |        | 39.3              | 52.4
[CrowS-Pairs]            | top-1  | 41.1              | 43.4
[BBQ Ambig][BBQ]         | top-1  | 62.6              | 71.1
[BBQ Disambig][BBQ]      | top-1  | 58.4              | 50.8
[Winogender]             | top-1  | 55.1              | 54.7
[TruthfulQA]             |        | 35.1              | 42.7
[Winobias 1_2][Winobias] |        | 58.4              | 56.4
[Winobias 2_2][Winobias] |        | 90.0              | 75.4
[Toxigen]                |        | 56.7              | 50.0

## Model usage and limitations

### Known limitations

These models have certain limitations that users should be aware of:

*   **Training data**
    *   The quality and diversity of the training data significantly influence
        the model's capabilities. Biases or gaps in the training data can lead
        to limitations in the model's responses.
    *   The scope of the training dataset determines the subject areas the model
        can handle effectively.
*   **Context and task complexity**
    *   LLMs are better at tasks that can be framed with clear prompts and
        instructions. Open-ended or highly complex tasks might be challenging.
    *   A model's performance can be influenced by the amount of context
        provided (longer context generally leads to better outputs, up to a
        certain point).
*   **Language ambiguity and nuance**
    *   Natural language is inherently complex. LLMs might struggle to grasp
        subtle nuances, sarcasm, or figurative language.
*   **Factual accuracy**
    *   LLMs generate responses based on information they learned from their
        training datasets, but they are not knowledge bases. They may generate
        incorrect or outdated factual statements.
*   **Common sense**
    *   LLMs rely on statistical patterns in language. They might lack the
        ability to apply common sense reasoning in certain situations.

### Ethical considerations and risks

The development of large language models (LLMs) raises several ethical concerns.
In creating an open model, we have carefully considered the following:

*   **Bias and fairness**
    *   LLMs trained on large-scale, real-world text data can reflect
        socio-cultural biases embedded in the training material. These models
        underwent careful scrutiny, input data pre-processing described and
        posterior evaluations reported in this card.
*   **Misinformation and misuse**
    *   LLMs can be misused to generate text that is false, misleading, or
        harmful.
    *   Guidelines are provided for responsible use with the model, see the
        [Responsible Generative AI
        Toolkit](https://ai.google.dev/gemma/responsible).
*   **Transparency and accountability**
    *   This model card summarizes details on the models' architecture,
        capabilities, limitations, and evaluation processes.
    *   A responsibly developed open model offers the opportunity to share
        innovation by making LLM technology accessible to developers and
        researchers across the AI ecosystem.

Risks Identified and Mitigations:

*   **Perpetuation of biases:** It's encouraged to perform continuous monitoring
    (using evaluation metrics, human review) and the exploration of de-biasing
    techniques during model training, fine-tuning, and other use cases.
*   **Generation of harmful content:** Mechanisms and guidelines for content
    safety are essential. Developers are encouraged to exercise caution and
    implement appropriate content safety safeguards based on their specific
    product policies and application use cases.
*   **Misuse for malicious purposes:** Technical limitations and developer and
    end-user education can help mitigate against malicious applications of LLMs.
    Educational resources and reporting mechanisms for users to flag misuse are
    provided. Prohibited uses of Gemma models are outlined in our [terms of
    use](https://www.kaggle.com/models/google/gemma/license/consent).
*   **Privacy violations:** Models were trained on data filtered for removal of
    PII (Personally Identifiable Information). Developers are encouraged to
    adhere to privacy regulations with privacy-preserving techniques.

## Intended usage

### Application

Open Large Language Models (LLMs) have a wide range of applications across
various industries and domains. The following list of potential uses is not
comprehensive. The purpose of this list is to provide contextual information
about the possible use-cases that the model creators considered as part of model
training and development.

*   **Content creation and communication**
    *   **Text generation:** These models can be used to generate creative text
        formats like poems, scripts, code, marketing copy, email drafts, etc.
    *   **Chatbots and conversational AI:** Power conversational interfaces for
        customer service, virtual assistants, or interactive applications.
    *   **Text summarization:** Generate concise summaries of a text corpus,
        research papers, or reports.
*   **Research and education**
    *   **Natural Language Processing (NLP) research:** These models can serve
        as a foundation for researchers to experiment with NLP techniques,
        develop algorithms, and contribute to the advancement of the field.
    *   **Language Learning Tools:** Support interactive language learning
        experiences, aiding in grammar correction or providing writing practice.
    *   **Knowledge Exploration:** Assist researchers in exploring large bodies
        of text by generating summaries or answering questions about specific
        topics.

### Benefits

At the time of release, this family of models provides high-performance open
large language model implementations designed from the ground up for Responsible
AI development compared to similarly sized models.

Using the benchmark evaluation metrics described in this document, these models
have shown to provide superior performance to other, comparably-sized open model
alternatives.

In particular, RecurrentGemma models achieve comparable performance to Gemma
models but are faster during inference and require less memory, especially on
long sequences.

[MMLU]: https://arxiv.org/abs/2009.03300
[HellaSwag]: https://arxiv.org/abs/1905.07830
[PIQA]: https://arxiv.org/abs/1911.11641
[SocialIQA]: https://arxiv.org/abs/1904.09728
[BoolQ]: https://arxiv.org/abs/1905.10044
[winogrande]: https://arxiv.org/abs/1907.10641
[CommonsenseQA]: https://arxiv.org/abs/1811.00937
[OpenBookQA]: https://arxiv.org/abs/1809.02789
[ARC-c]: https://arxiv.org/abs/1911.01547
[TriviaQA]: https://arxiv.org/abs/1705.03551
[Natural Questions]: https://github.com/google-research-datasets/natural-questions
[HumanEval]: https://arxiv.org/abs/2107.03374
[MBPP]: https://arxiv.org/abs/2108.07732
[GSM8K]: https://arxiv.org/abs/2110.14168
[MATH]: https://arxiv.org/abs/2103.03874
[AGIEval]: https://arxiv.org/abs/2304.06364
[BIG-Bench]: https://arxiv.org/abs/2206.04615
[RealToxicity]: https://arxiv.org/abs/2009.11462
[BOLD]: https://arxiv.org/abs/2101.11718
[CrowS-Pairs]: https://aclanthology.org/2020.emnlp-main.154/
[BBQ]: https://arxiv.org/abs/2110.08193v2
[Winogender]: https://arxiv.org/abs/1804.09301
[TruthfulQA]: https://arxiv.org/abs/2109.07958
[winobias]: https://arxiv.org/abs/1804.06876
[Toxigen]: https://arxiv.org/abs/2203.09509
