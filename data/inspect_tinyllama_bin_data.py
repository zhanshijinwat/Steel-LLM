HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24
import struct
import numpy as np
from transformers import AutoTokenizer

dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}
def read_header(path):
    with open(path, "rb") as f:
        magic = f.read(len(HDR_MAGIC))
        assert magic == HDR_MAGIC, "File doesn't match expected format."
        version = struct.unpack("<Q", f.read(8))
        assert version == (1,)
        (dtype_code,) = struct.unpack("<B", f.read(1))
        dtype = dtypes[dtype_code]
        (chunk_size,) = struct.unpack("<Q", f.read(8))
    return dtype, chunk_size

if __name__ == "__main__":
    bin_data_dir = "/data1/step3_final_data/chat/chat_process0_0000000001.bin"
    tokenizer = AutoTokenizer.from_pretrained("../model/qwen_moe")
    dtype, chunk_size = read_header(bin_data_dir)
    print(dtype, chunk_size)
    mmap = np.memmap(bin_data_dir, mode="r", order="C", offset=HDR_SIZE)
    print(mmap, len(mmap))
    view = memoryview(mmap)
    print(view)
    arr1 = np.frombuffer(view, dtype=dtype, count=2049, offset=0)
    # 偏移量需要按字节来
    arr2 = np.frombuffer(view, dtype=dtype, count=2049, offset=np.dtype(dtype).itemsize*1)
    print(arr1[:10])
    print(arr2[:10])
    text = tokenizer.decode(arr1)
    print("len: ", len(arr1), len(text), np.dtype(dtype).itemsize)