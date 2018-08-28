import subprocess

# fxn taken from https://discuss.pytorch.org/t/memory-leaks-in-trans-conv/12492

def get_gpu_memory_map():   
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    
    return float(result)