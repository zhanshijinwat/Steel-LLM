import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import os
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import Block, Config, CausalSelfAttention
from transformers import AutoConfig, AutoModelForCausalLM
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset, create_dataloader
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from lit_gpt.triton_cross_entropy import TritonCrossEntropyLoss
import random
from loguru import logger
# os.environ["WANDB_MODE"]="offline"

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
# test train code model: qwen1
# sys.path.append(os.path.join(parent_dir, "model", "qwen_1_1_8B_chat"))
# from modeling_qwen import QWenLMHeadModel, QWenBlock
# recurrentgemma: too slow
# sys.path.append(os.path.join(parent_dir, "model", "recurrentgemma"))
# from modeling_recurrent_gemma import RecurrentGemmaForCausalLM, RecurrentGemmaDecoderLayer
# Steel LLM model
sys.path.append(os.path.join(parent_dir, "model", "steel_modify_from_qwen_1_5"))
from modeling_steel import SteelForCausalLM, SteelDecoderLayer
from steel_llm_utils import compatible_tiny_llama_config

import logging
logging.basicConfig(level=logging.DEBUG)

# model_name = "steel_llm_test_qwen1"
name = "steel_llm"
out_dir = Path("/data/gu_data/ckpt") / name
TRAIN_DATA_DIR = Path("/data/gu_data/step3_input")
# TRAIN_DATA_DIR = Path("/data/step3_train_input/test")
MODEL_PATH = "../model/steel_modify_from_qwen_1_5"
# todo: check block size
BLOCK_SIZE = 2048
# bool / Path
# resume model and data
RESUME = False
# RESUME = Path("/home/calfa100/gqs/Steel-LLM/pretrain_modify_from_TinyLlama/pretrain/out/steel_llm/step-200000-iter-1600000-ckpt")
ONLY_RESUME_MODEL =  Path("/home/ubuntu/gu/step-220000-iter-1760000-ckpt")
ADD_NEW_DATA_DIR = None
# qwen moe pad_token_id
IGNORE_INDEX = 151643
USE_FLASH_ATTN =True # "auto"

# Hyperparameters
num_of_devices = 8
global_batch_size = 64*num_of_devices
learning_rate = 3e-4
micro_batch_size = 8
# cal step 1: 1640*10**9/4/2048/512 
# cal day 1: 1800*10**9/4/2048/8/1.4/8/3600/24
# cal day2: 1640*10**9/4/8/23800/3600/24
max_step = 430000 * 2
# lr scheduler
decay_lr = True
lr_decay_step = int(max_step*0.9)
min_lr = 3e-5
warmup_steps = 1_000
#---
log_step_interval = 20
eval_iters = 100 # eval iter
save_step_interval = 20000
eval_step_interval = 20000
 

weight_decay = 0.05
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps


max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = lr_decay_step * gradient_accumulation_steps
log_iter_interval = log_step_interval * gradient_accumulation_steps


# Treat all dataset equally by their size. If you want to use a different weight for a dataset, add it to the list with the weight.
# 数据根目录下的文文件名开头，""为匹配所有文件
train_data_config = [
    ("", 1),
]

val_data_config = [
    ("validation", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", name, flush_logs_every_n_steps=log_iter_interval)
wandb_logger = WandbLogger()


def setup(
    devices: int = num_of_devices,
    train_data_dir: Path = TRAIN_DATA_DIR,
    val_data_dir: Optional[Path] = None,
    precision: Optional[str] = None,
    resume: Union[bool, Path] = RESUME,
    model_path = MODEL_PATH,
    # num/None
    block_size = BLOCK_SIZE,
    only_resume_model = ONLY_RESUME_MODEL,
) -> None:
    precision = precision or get_default_supported_precision(training=True, tpu=False)
    print(precision)
    config = AutoConfig.from_pretrained(model_path,trust_remote_code=True)
    print(config)
    if devices > 1: 
        # todo: check param
        strategy = FSDPStrategy(
            # sharding_strategy = "SHARD_GRAD_OP",
            auto_wrap_policy={SteelDecoderLayer},
            activation_checkpointing_policy=None,
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    # 添加其他参数
    config.model_path = model_path
    config.use_flash_attn = USE_FLASH_ATTN
    config = compatible_tiny_llama_config(config, block_size)
    # todo: why not use?
    if devices > 1:
        fabric.launch(main, train_data_dir, val_data_dir, resume, config, only_resume_model)
    else:
        main(fabric, train_data_dir, val_data_dir, resume, config, only_resume_model)


def main(fabric, train_data_dir, val_data_dir, resume, config, only_resume_model):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)


    train_dataloader, val_dataloader, train_datasets, val_datasets = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
        train_data_config = train_data_config,
        val_data_config = val_data_config
    )
    print("finish load data index...") 
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = SteelForCausalLM(config)
        # print(model.transformer.wte.weight)
        model.apply(model._init_weights) 
        # print(model.transformer.wte.weight)

    # 预估参数量和计算量
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")
    with torch.device("meta"):
        meta_model = SteelForCausalLM(config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        config.estimated_flops = estimated_flops
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, config.block_size))
        
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x
        
    # if resume:
    #     del model
    #     model_path = resume / "huggingface-ckpt"
    #     model = QWenLMHeadModel.from_pretrained(model_path)
    #     fabric.print(f"Resuming model from {resume} success...")

    huggingface_format_model = model
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    ) 
    
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model,"optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}
    if resume or only_resume_model:
        if resume != False:
            assert(only_resume_model==False)
            state_dir = resume / "state.pth"
        else:
            assert(resume==False)
            state_dir = only_resume_model / "state.pth"
        fabric.load(state_dir, state)
        fabric.print(f"Resuming state from {resume} success...")

    state["model"] = model


    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume, config, train_datasets, huggingface_format_model)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume, config, train_datasets, huggingface_format_model):
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        validate(fabric, model, val_dataloader)  # sanity check


    total_lengths = 0
    total_t0 = time.perf_counter()
    
    initial_iter = state["iter_num"]
    loss_func = TritonCrossEntropyLoss(ignore_index = IGNORE_INDEX)
    fabric.print(f"start state: {state}") 
    for  train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if resume:
            resume = False
            fabric.barrier()
            fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break
        
        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            output = model(input_ids)
            logits = output.logits
            loss = loss_func(logits, targets)
            # loss = torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]), targets.reshape(-1)) # raw torch loss
            # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        
        state["iter_num"] += 1
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
                f"data idx: {train_datasets[0].iterator._curr_idx}/{len(train_datasets[0].iterator._block_idxs)}"
                f"lr: {lr}"
            )
 
        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=config.estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        ) 
            
        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": val_loss.item(), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item()), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.barrier()

        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir / f"step-{state['step_count']:06d}-iter-{state['iter_num']:06d}-ckpt"
            if not os.path.exists(checkpoint_path) and fabric.global_rank==0:
                os.makedirs(checkpoint_path, exist_ok=True)
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            # fabric control only save one copy
            fabric.save(os.path.join(checkpoint_path, "state.pth"), state)
            # train_datasets[0].iterator.save_param(os.path.join(checkpoint_path, f"data-state-rank-{fabric.global_rank}.json"))
            train_datasets[0].iterator.save_pikle(os.path.join(checkpoint_path, f"data-state-rank-{fabric.global_rank}.pikle"))
            logging.info(f"save train_datasets success...")
            # only worker 0 save ckpt. use fabric while be stuck...
            # if fabric.global_rank == 0:
            #     print(huggingface_format_model)
            #     print(type(huggingface_format_model), type(model))
            #     huggingface_format_model.save_pretrained(os.path.join(checkpoint_path, f"huggingface-ckpt"))
            fabric.barrier()
        
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        #loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        loss_func = TritonCrossEntropyLoss()
        loss = loss_func(logits, targets)
        losses[k] = loss.item()
        
    out = losses.mean()

    model.train()
    return out


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
    train_data_config = None,
    val_data_config = None
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    if RESUME: 
        packdata_ckpt_dir = str(RESUME / f"data-state-rank-{fabric.global_rank}.pikle")
    else: 
        packdata_ckpt_dir = None 
    train_dataloader, train_datasets = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train",
        train_data_config = train_data_config,
        packdata_ckpt_dir = packdata_ckpt_dir,
        add_new_data_dir = ADD_NEW_DATA_DIR
    )
    if val_data_dir:
        val_dataloader, val_datasets = create_dataloader(
                batch_size=batch_size,
                block_size=effective_block_size,
                fabric=fabric,
                data_dir=val_data_dir,
                shuffle=False,
                seed=seed,
                split="validation",
                val_data_config = val_data_config
            )
    else:
        val_dataloader, val_datasets = None, None
            
    return train_dataloader, val_dataloader, train_datasets, val_datasets


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
