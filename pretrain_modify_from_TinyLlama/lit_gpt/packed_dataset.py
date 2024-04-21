# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import os
import random
import struct
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import glob
import lightning as L
import logging
import  hashlib
import time
import json
import pickle
from copy import deepcopy

dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def code(dtype):
    for k in dtypes:
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    def __init__(
        self, hash_file_map, block_size, seed=12345, shuffle=True, wrap=False, num_processes=1, process_rank=0,
        packdata_ckpt_dir = None, add_data_hash_file_map=None
    ):
        self._hash_file_map = hash_file_map
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank
        self.packdata_ckpt_dir = packdata_ckpt_dir
        self.add_data_hash_file_map = add_data_hash_file_map

    def __iter__(self):
        # None
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id
        # 不需要max_num_files, 某个worker迭代完了会重新刷数据
        max_num_files = len(self._hash_file_map) // num_shards * num_shards
        logging.debug(f"worker_info: {worker_info}, num_workers:{num_workers}, worker_id:{worker_id}, \
                      num_shards:{num_shards}, shard_id:{shard_id}, max_num_files:{max_num_files}")
        
        # 不同worker处理不通的文件
        worker_hash_list = list(self._hash_file_map.keys())[shard_id::num_shards] 
        worker_hash_file_map = {}
        for hash in worker_hash_list:
            worker_hash_file_map[hash] = self._hash_file_map[hash]

        # 处理add data
        worker_add_data_hash_file_map = None
        if self.add_data_hash_file_map is not None:
            worker_add_data_hash_list = list(self.add_data_hash_file_map.keys())[shard_id::num_shards] 
            worker_add_data_hash_file_map = {}
            for hash in worker_add_data_hash_list:
                worker_add_data_hash_file_map[hash] = self.add_data_hash_file_map[hash]
            if len(worker_add_data_hash_file_map) == 0:
                worker_add_data_hash_file_map = None
            logging.debug(f"worker_add_data_hash_file_map:{worker_add_data_hash_file_map} \n \
                      len of worker_add_data_hash_file_map: {len(worker_add_data_hash_file_map)}")
        logging.info(f"PackedDataset __iter__: \n {worker_add_data_hash_file_map}")
 
        
        logging.debug(f"worker_hash_file_map:{worker_hash_file_map} \n len of worker_hash_file_map: {len(worker_hash_file_map)}")

        # 每个worker的seed是不同的
        self.iterator = PackedDatasetIterator(
            hash_file_map=worker_hash_file_map,
            n_chunks=len(worker_hash_file_map),
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
            packdata_ckpt_dir = self.packdata_ckpt_dir,
            add_data_hash_file_map = worker_add_data_hash_file_map,
            
        )
        return self.iterator


class PackedDatasetBuilder(object):
    def __init__(self, outdir, prefix, chunk_size, sep_token, dtype="auto", vocab_size=None):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames = []
        self.all_tokens = 0

    def _write_chunk(self):
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)
        print(f"chunk write to {filename}")

        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        self._filenames.append(filename)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr):
        # 用while防止一个句子是好几个chunk的长度
        self.all_tokens += arr.shape[0]
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_reminder(self):
        print(f"{self._prefix} remind {self._idx} part size{self._chunk_size} used {self._idx/self._chunk_size}")
        self._write_chunk()


class PackedDatasetIterator:
    def __init__(self, hash_file_map, n_chunks, block_size, seed, shuffle, wrap, 
                 packdata_ckpt_dir = None, add_data_hash_file_map = None):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._wrap = wrap

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.
        self._hash_file_map = hash_file_map
        self._filenames = list(hash_file_map.values())
        logging.debug(f"PackedDatasetIterator: hash map: {self._hash_file_map} \n files: {self._filenames}")
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None

        self._mmaps = []
        self._buffers = []

        self._block_idxs = []
        self._curr_idx = 0
        self.file_end = False

        if packdata_ckpt_dir == None:
            assert(add_data_hash_file_map is None)
            self._load_n_chunks()
        else:
            self.load_pikle(load_dir=packdata_ckpt_dir)
            logging.info(f"add_data_hash_file_map: {add_data_hash_file_map}")
            if add_data_hash_file_map is not None:
                self.add_new_data(add_data_hash_file_map) 
        logging.info(f"PackedDatasetIterator: \n {self.__dict__}") 

    def save_param(self, save_dir):
        save_dict = {}
        save_dict["_hash_file_map"] = self._hash_file_map
        save_dict["_file_idx"] = self._file_idx
        # fix
        save_dict["_n_chunks"] = self._n_chunks
        # fix
        save_dict["_block_size"] = self._block_size
        # fix 
        save_dict["_n_blocks"] = self._n_blocks
        save_dict["_curr_idx"] = self._curr_idx
        # fix
        save_dict["n_all_blocks"] = len(self._block_idxs)
        save_dict["_block_idxs"] = self._block_idxs.tolist()
        save_dict["_rng_state"] = self._rng.bit_generator.state
        with open(save_dir, 'w') as file:
            json.dump(save_dict, file)
    
    def save_pikle(self, save_dir):
        save_data = {}
        for k,v in self.__dict__.items():
            # Variable cannot be serialized
            if k in ["_mmaps", "_buffers"]:
                continue
            save_data[k] = deepcopy(self.__dict__[k])
        print(save_data)
        with open(save_dir, 'wb') as file:
            pickle.dump(save_data, file)
        del save_data

    def load_pikle(self, load_dir):
        logging.info(f"load packdata from {load_dir}......")
        with open(load_dir, 'rb') as file:
            data = pickle.load(file)
            self.__dict__.update(data)
        # realod buffer and mmap
        self._file_idx -= self._n_chunks
        self.build_mem_map_and_buffer(check_hash=True)
        self._file_idx += self._n_chunks
        logging.debug(f"load packdata param: \n {self.__dict__}")

    def add_new_data(self, add_data_hash_file_map):
        logging.info(f"add new data before: all blocks={len(self._block_idxs)}, \
                     cur_idx={self._curr_idx} filidx={self._file_idx} filenum={len(self._hash_file_map)} all_n_chunks={self._n_chunks}")
        if self._n_chunks != len(self._hash_file_map):
            raise NameError("add_new_data must all data shuffle")
        # 过滤重复数据
        new_data_set = set(add_data_hash_file_map.keys())
        old_data_set = set(self._hash_file_map.keys())
        dup_set = old_data_set.intersection(new_data_set)
        if len(dup_set) != 0:
            raise NameError("add repeated data")
        print(f"add_new_data self.__dict__: \n {self.__dict__}")
        # 控制epoch的idx
        self._file_idx += len(add_data_hash_file_map)
        for k,v in add_data_hash_file_map.items():
            self._hash_file_map[k] = v
        self._filenames = list(self._hash_file_map.values())
        self._n_chunks += len(add_data_hash_file_map)
        #
        new_all_file_blocks = 0
        for hash, file_dir in add_data_hash_file_map.items():
            dtype, chunk_size = self._read_header(file_dir)
            assert(chunk_size%self._block_size==0)
            assert(dtype==self._dtype and (chunk_size//self._block_size)==self._n_blocks)
            mmap = np.memmap(file_dir, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))
            new_all_file_blocks += (chunk_size//self._block_size)
        new_data_idxs = self._rng.permutation(new_all_file_blocks) if self._shuffle else range(new_all_file_blocks)
        # add bias
        new_data_idxs = new_data_idxs + len(self._block_idxs)
        self._block_idxs = np.append(self._block_idxs, new_data_idxs)
        # shuffle not use data idx
        need_shuffle_idxs = self._block_idxs[self._curr_idx:]
        self._rng.shuffle(need_shuffle_idxs)
        self._block_idxs[self._curr_idx:] = need_shuffle_idxs
        logging.info(f"add new data after: all blocks={len(self._block_idxs)}, \
                     cur_idx={self._curr_idx} filidx={self._file_idx} filenum={len(self._hash_file_map)} all_n_chunks={self._n_chunks}") 

    
    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()

    
    def build_mem_map_and_buffer(self, check_hash = False):
        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]
            if check_hash:
                new_hash = CalcMD5(filename)
                if new_hash not in self._hash_file_map or self._hash_file_map[new_hash]!=filename:
                    raise NameError(f"new hash diff with old hash, file changed...")
            # 必须保证每个chunk shape大小是固定的
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(filename)
                self._n_blocks = self._chunk_size // self._block_size
                if self._chunk_size % self._block_size != 0:
                    raise NameError(f"block size error: {self._block_size} {self._chunk_size}")
            else:
                dtype, chunk_size = self._read_header(filename)
                assert(dtype==self._dtype and (chunk_size//self._block_size)==self._n_blocks)

            # TODO: check header matches with previous files
            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))
    
    def _load_n_chunks(self):
        """build index from raw"""
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        assert(self._n_chunks <= len(self._filenames))
        # 无限循环读取
        if self._n_chunks > len(self._filenames[self._file_idx :]): 
            # if not self._wrap:
            #     raise StopIteration
            logging.info(f"1 epoch finish. file num:{len(self._filenames)}, file_idx:{self._file_idx }, n_chunk:{self._n_chunks}")
            if len(self._filenames[self._file_idx :])!=0 and self.file_end == False:
                self._file_idx = len(self._filenames)-self._n_chunks
                self.file_end = True
            else:    
                self._file_idx = 0
                self.file_end = False

        self.build_mem_map_and_buffer(check_hash=False)

        self._file_idx += self._n_chunks
        # n个chuncks里边一共有多少个句子
        n_all_blocks = self._n_chunks * self._n_blocks

        # 需要保持随机数状态
        self._block_idxs = self._rng.permutation(n_all_blocks) if self._shuffle else range(n_all_blocks)

        self._curr_idx = 0

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
        self._curr_idx += 1
        return torch.from_numpy(arr.astype(np.int64))


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)

def CalcMD5(filepath, limit_size = 50*(10**6)):
    """避免文件太大hash时间太长"""
    size = os.path.getsize(filepath)
    t0 = time.time()
    with open(filepath,'rb') as f:
        md5obj = hashlib.md5()
        if size < limit_size:
            md5obj.update(f.read())
        else:
            md5obj.update(f.read(limit_size//2))
            f.seek(-limit_size//2, os.SEEK_END)
            md5obj.update(f.read(limit_size//2))
        hash = md5obj.hexdigest()
        logging.debug(f"CalcMD5 cost {time.time()-t0}s")
        return hash

def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train",
    train_data_config = None, val_data_config=None, packdata_ckpt_dir = None, add_new_data_dir = None
) -> DataLoader:
    if add_new_data_dir!=None and packdata_ckpt_dir==None:
        raise NameError("add new data must use load ckpt, if you train from raw, add data directly.")
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config

    for prefix, _ in data_config:
        hash_file_map = {}
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        logging.info("create_dataloader root path: ",str(data_dir / f"{prefix}*"))
        # 计算文件hash值
        for f in filenames:
            hash = CalcMD5(f)
            if hash in hash_file_map:
                logging.warning(f"hash collide: {f}")
            hash_file_map[hash] = f
        hash_file_map = hash_file_map.items()
        hash_file_map = {k: v for k, v in sorted(hash_file_map)}
        logging.info(f"hash_file_map: {hash_file_map} \n hash_file_map len:{len(hash_file_map)}")

        
        add_data_hash_file_map = None
        # find and add data
        if add_new_data_dir != None:
            add_data_hash_file_map = {}
            add_data_filenames = sorted(glob.glob(str(add_new_data_dir / f"{prefix}*")))
            for f in add_data_filenames:
                hash = CalcMD5(f)
                if hash in add_data_hash_file_map:
                    logging.warning(f"hash collide: {f}")
                add_data_hash_file_map[hash] = f
            add_data_hash_file_map = add_data_hash_file_map.items()
            add_data_hash_file_map = {k: v for k, v in sorted(add_data_hash_file_map)}
            logging.info(f"add_data_hash_file_map: {add_data_hash_file_map}") 


        dataset = PackedDataset(
            hash_file_map,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            # n_chunks=len(hash_file_map),
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
            packdata_ckpt_dir = packdata_ckpt_dir,
            add_data_hash_file_map = add_data_hash_file_map
        )
        datasets.append(dataset)

        # add new data only on first dataset.
        add_new_data_dir = None
    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True), datasets

def test_create_dataloader():
    logging.basicConfig(level=logging.DEBUG)
    seed = 666
    data_config = [("", 1)]
    class Param():
        def __init__(self) -> None:
            self.global_rank =3
            self.world_size = 8
    fabric = Param()
    dataloader,datasets = create_dataloader(
        batch_size=4,
        block_size=2049,
        fabric=fabric,
        data_dir=Path("/data/step3_train_input/test"),
        shuffle=True,
        seed=seed,
        split="train",
        train_data_config = data_config,
        packdata_ckpt_dir = "data_state-100.pikle",
        add_new_data_dir = Path("/data/step3_train_input/test_add")
    )
    # copy_dataloader = iter(dataloader)
    counter = 0
    load = True
    for data in dataloader:
        print(f"{counter}: ",data[0,-10:].tolist())
        # print(next(copy_dataloader)[0,0:10])
        counter += 1
        if counter % 100 ==0:
            datasets[0].iterator.save_param(f"data_state-{counter}.json")
            datasets[0].iterator.save_pikle(f"data_state-{counter}.pikle")

if __name__ == "__main__":
    test_create_dataloader()
