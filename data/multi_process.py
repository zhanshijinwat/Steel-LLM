import multiprocessing as mp
import concurrent
from multiprocessing import get_context
import concurrent.futures
import pandas as pd
import time

class MultiprocessPoolTool():
    def __init__(self, max_workers=16):
        self.max_workers = max_workers

    def run_iter_file(self, func, file_list, func_arg_list = []):
        """多进程遍历文件处理数据,file_list为文件路径，func_arg_list里为处理超参数"""
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers,mp_context=get_context("spawn"))
        self.workers = []
        for f in file_list:
            new_func_arg_list = [f, *func_arg_list]
            self.workers.append(
                self.pool.submit(func, *new_func_arg_list)
            )
        for future in concurrent.futures.as_completed(self.workers):
            pass
        self.stop()

    def stop(self,):
        for w in self.workers:
            w.cancel()
        self.pool.shutdown(wait=True)

def add(file_name, a,b):
    try:
        pd.DataFrame([a+b]).to_csv(file_name)
        print(file_name, a+b)
        return a+b
    except Exception as e:
        print(e)

def MultiprocessFunc():
    # 定义要执行的函数
    def worker(arg1, t):
        time.sleep(3)
        print(f"{arg1}, {time.time()-t}")
    process = []
    t1 = time.time()
    for i in range(5):
        process.append(mp.Process(target=worker, args=(i,t1)))  
    [p.start() for p in process]
    [p.join() for p in process]

if __name__ == "__main__":
    # obj = MultiprocessPoolTool()
    # obj.run_iter_file(func=add, file_list = ["file1.np", "file2.np", "file3.np"], func_arg_list = [6,3])
    MultiprocessFunc()

