import multiprocessing as mp
import concurrent
from multiprocessing import get_context
import concurrent.futures
import pandas as pd

class MultiprocessTool():
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
    except:
        

if __name__ == "__main__":
    obj = MultiprocessTool()
    obj.run_iter_file(func=add, file_list = ["file1.np", "file2.np", "file3.np"], func_arg_list = [6,3])

