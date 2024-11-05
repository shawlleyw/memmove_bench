import torch
import time
from argparse import ArgumentParser
from memmove.torch_op import permute_tokens as torch_move
from memmove.triton_op import permute_tokens as triton_move

class Benchmark:
    
    def __init__(self, name, op, warmup_iter=5, run_iter=10):
        self.op = op
        self.warmups = warmup_iter
        self.runs = run_iter
        self.name = name
        
    def _warm_up(self, *args):
        for i in range(self.warmups):
            self.op(*args)
            
    def run(self, *args):
        self._warm_up(*args)
        torch.cuda.synchronize()
        start = time.time()
        
        for i in range(self.runs):
            self.op(*args)
            
        torch.cuda.synchronize()
        end = time.time()
        
        elapse = (end - start) * (10 ** 6)
        print(f"benchmark {self.name} takes: {elapse:.2f} us")
        
    def __call__(self, *args):
        self.run(*args)
        

def get_args():
    parser = ArgumentParser()
    
    parser.add_argument("-d", "--dim", type=int, default=4096)
    parser.add_argument("-b", "--batch", type=int, default=64)
    parser.add_argument("-e", "--experts", type=int, default=None)
    
    args = parser.parse_args()
    return args

@torch.inference_mode
def main():
    args = get_args()
    
    if not args.experts:
        # generate mappings
        mappings = torch.randperm(args.batch, device="cuda:0")
    else:
        # generate expert id (bin index)
        pass
    
    
    inputs = torch.randn((args.batch, args.dim), dtype=torch.bfloat16, device="cuda:0")
    print(f"Input shape: {inputs.shape}, dtype={inputs.dtype}")
    
    torchop = Benchmark("torch", torch_move)
    tritonop = Benchmark("triton", triton_move)
    
    torchop(inputs, mappings)
    tritonop(inputs, mappings)
        
    


if __name__ == '__main__':
    main()