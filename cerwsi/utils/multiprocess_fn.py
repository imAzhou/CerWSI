import multiprocessing
from multiprocessing import Pool
import numpy as np

def multiprocess_fn(process_fn, process_datas, cpu_num, **args):
    set_split = np.array_split(process_datas, cpu_num)
    print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
    workers = Pool(processes=cpu_num)
    processes = []
    for proc_id, set_group in enumerate(set_split):
        p = workers.apply_async(process_fn, (proc_id, set_group, args))
        processes.append(p)

    total_results = []
    for p in processes:
        total_results.append(p.get())
    
    return total_results