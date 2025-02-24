
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from vllm import LLM, SamplingParams
import re
from math_verify import parse, verify
import json
from tqdm import tqdm
def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str) # extract the solution after ####
    assert solution is not None
    # #### 72
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

if __name__ == "__main__":
    model_dir = "/xxx/gsm8k_function_rm_10epoch_roll16/global_step_450/actor/huggingface"
    gsmk8k_traindata_dir = "/xxx/gsm8k/main/train-00000-of-00001.parquet"
    SAMPLE_NUM = 5
    INFER_QUESTION_NUM = 20
    output_dir = 'distill_data_from_r1.jsonl'
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 设置采样参数，如温度、核采样参数、重复惩罚、最大生成长度等
    sampling_params = SamplingParams(temperature=1, top_p=0.9, repetition_penalty=1.05, max_tokens=2048)

    # 指定已下载的模型存储路径
    llm = LLM(model=model_dir)

    all_data = pd.read_parquet(gsmk8k_traindata_dir)
    save_data = []
    # 准备提示词
    prompt_cache_dict = {}
    answer_cache_dict = {}
    message_cache = []
    message_idx_to_question_idx = {}
    for i in tqdm(range(len(all_data))):
        question = all_data["question"].iloc[i]
        answer = all_data["answer"].iloc[i]
        try:
            solution = extract_solution(answer)
        except:
            print(f'[no solution] idx:{i}')
            continue
        prompt = question + ' ' + "Let's think step by step and output the final answer after \"####\"."
        raw_prompt = prompt
        prompt_cache_dict[i] = raw_prompt
        answer_cache_dict[i] = solution
        messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        for s in range(SAMPLE_NUM):
            message_cache.append(text)
            message_idx_to_question_idx[len(message_cache) - 1] = i

        # 攒batch加速推理
        if (i+1) % INFER_QUESTION_NUM == 0:
            # 生成输出
            print(f"infer num:{len(message_cache)} {(i+1)%INFER_QUESTION_NUM}")
            outputs = llm.generate(message_cache, sampling_params)
            # 打印输出
            # print("prompt1:", texts[0])
            for j, output in enumerate(outputs):
                a_data = {}
                prompt = output.prompt
                generated_text = output.outputs[0].text
                try:
                    predict = extract_solution(generated_text)
                except:
                    print(f"no solution in generated_text idx:{i}")
                    continue
                question_idx = message_idx_to_question_idx[j]
                raw_prompt = prompt_cache_dict[question_idx]
                solution = answer_cache_dict[question_idx]
                a_data["idx"] = question_idx
                a_data["prompt"] = raw_prompt
                a_data["generated_text"] = generated_text
                a_data["predict"] = predict
                a_data["answer"] = solution
                predict = parse(predict)
                answer  = parse(solution)
                right = verify(predict, answer)
                a_data["if_right"] = right
                
                with open(output_dir, 'a', encoding='utf-8') as file:
                    json_str = json.dumps(a_data, ensure_ascii=False)
                    file.write(json_str + "\n")
            prompt_cache_dict = {}
            answer_cache_dict = {}
            message_cache = []
            message_idx_to_question_idx = {}