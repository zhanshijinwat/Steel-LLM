import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Config, Tokenizer
from transformers import AutoTokenizer
import time

filenames_sample = []
# key: name value: dir
filename_sets = {
    "sky": "data/sky/data/*jsonl",
}


def prepare_sample(
    source_path: Path, checkpoint_dir: Path, destination_path: Path, chunk_size: int, match: str = ""
) -> None:
    raise NotImplementedError("not modify prepare_sample...")
    """Prepare the "Red Pajama" dataset using the original tokenizer."""
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    for name in filenames_sample:
        if match and match not in name:
            continue

        filepath = source_path / name

        if not filepath.is_file():
            raise RuntimeError(
                f"Input file not found at {filepath}. \nMake sure you download the data, e.g. wget -i"
                " https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt or through"
                " \nhttps://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T"
                " \nhttps://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample \n"
            )

        prefix, _ = os.path.splitext(name)

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        print(f"Processing {name}")

        with open(filepath, encoding="utf-8") as f:
            for row in tqdm(f):
                text = json.loads(row)["text"]
                text_ids = tokenizer.encode(text)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()

def process_line_text(text, tokenizer):
    t1 = time.time()
    text_ids = tokenizer.encode(text)
    print(time.time() - t1)
    return text_ids

def process_sky_file(file_dir, builder, tokenizer):
    SAVE_NUM = 10000
    cache_text = ""
    counter = 0
    with open(file_dir, encoding="utf-8") as f:
        for row in tqdm(f):
            counter += 1
            text = json.loads(row)["text"]
            text += "<|im_end|>"
            cache_text += text
            if counter%SAVE_NUM==0:
                text_ids = process_line_text(text=cache_text, tokenizer=tokenizer)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
                cache_text = ""
                raise 

        if cache_text!="":
            text_ids = process_line_text(text=cache_text, tokenizer=tokenizer)
            builder.add_array(np.array(text_ids, dtype=builder.dtype))

def prepare_full(
    source_path: Path, checkpoint_dir: Path, destination_path: Path, chunk_size: int, match: str = "", max_files = 100000000000000,
) -> None:
    """
    chunk_size: chunk_size=(block_size + 1) * 1024 （block是序列长度）
    """
    import zstandard as zstd

    destination_path.mkdir(parents=True, exist_ok=True)

    # tokenizer = Tokenizer(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer.parallelism = 1
    # "arxiv": "arxiv/arxiv*"
    for set_name, pattern in filename_sets.items():
        # 循环不同的数据来源
        if match and match not in set_name:
            raise NameError("should not skip...")
            continue

        filenames = glob.glob(os.path.join(source_path, pattern), recursive=True)
        filenames = filenames[:max_files]
        print(filenames)
        print(f"{set_name} have { len(filenames)} files...")
        if not filenames:
            raise RuntimeError(f"No files matching {pattern} found at {source_path}.")

        # 不太好用多进程，会涉及到共享part的idx
        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=set_name,
            chunk_size=chunk_size,
            # 填充token，用pad，而不是用eos_id
            sep_token=tokenizer.pad_token_id,
            dtype="auto",
            vocab_size=len(tokenizer),
        )

        for name in tqdm(filenames):
            filepath = source_path / name
            print(f"Processing {name}...")
            if set_name == "sky":
                process_sky_file(file_dir=filepath, builder=builder,tokenizer=tokenizer)
            else:
                raise NameError(f"not have this set {set_name}")
            print(f"all chunks: {builder._counter}  all tokens: {builder.all_tokens}...")
        # 将cache里边剩余的内容写入
        builder.write_reminder()


def prepare(
    source_path: Path = Path("/"),
    # tokenizer地址
    checkpoint_dir: Path = Path("/hoe/test/gqs/Steel-LLM/model/qwen_moe"),
    destination_path: Path = Path("/data/pretrain_input/sky"),
    sample: bool = False,
    match: str = "",
    max_files = 1,
    # 序列长度
    block_size = 2048,
    # 一个数据分片有多少个序列
    # 1024一个chunk是8M 
    blocks_in_a_chunck = 1024*100


) -> None:
    """Prepare the SteelLLM dataset. We assume tokenizer has been trained."""
    prepare_fn = prepare_sample if sample else prepare_full
    prepare_fn(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path,
        # 
        chunk_size=(block_size + 1) * blocks_in_a_chunck,  # block size + 1 for causal, 1024 blocks
        match=match,
        max_files = max_files,
    )


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)