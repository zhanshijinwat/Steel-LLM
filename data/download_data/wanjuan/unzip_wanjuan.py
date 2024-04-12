import glob 
import os
import time
if __name__ == "__main__":
    dir_list = glob.glob("/data/step0_raw/wanjuan/**/*.jsonl.tar.gz", recursive=True)
    target_root_dir = "/data/step2_data_cleaning/wanjuan"
    print(dir_list)
    print(len(dir_list))
    for coutner, file_dir in enumerate(dir_list):
        print(file_dir)
        t0 = time.time()
        save_need_name = file_dir.split("/")[-3:]
        tmp_name = save_need_name[-1]
        # 解压后的文件名称
        tmp_name = tmp_name.replace(".jsonl.tar.gz", ".jsonl")
        tmp_name = os.path.join(target_root_dir, tmp_name)
        # 目标名称带着文件夹名
        save_need_name[-1] = save_need_name[-1].split(".")[0]
        target_file_name = "_".join(save_need_name)
        target_file_name += ".jsonl"
        target_file_name = os.path.join(target_root_dir, target_file_name)
        print("old_name:", tmp_name)
        print("new name:",target_file_name)
        if os.path.exists(target_file_name):
            print(f"have: {target_file_name}")
        cmd = f"tar -zxvf {file_dir} -C {target_root_dir}"
        os.system(cmd)
        os.system(f"mv {tmp_name} {target_file_name}")
        print(coutner, time.time()-t0)
    
    en_dir = os.path.join(target_root_dir, "wanjuan_en")
    cn_dir = os.path.join(target_root_dir, "wanjuan_zh")
    os.mkdir(en_dir)
    os.mkdir(cn_dir)
    mv_cmd1 = f"mv {target_root_dir}/EN*.jsonl {en_dir}"
    os.system(mv_cmd1)
    mv_cmd2 = f"mv {target_root_dir}/CN*.jsonl {cn_dir}"
    os.system(mv_cmd2)
