import torch
import time
from argparse import ArgumentParser
from memmove.torch_op import permute_tokens as torch_move, gpu_to_cpu, cpu_to_gpu
from memmove.triton_op import permute_tokens as triton_move
from memmove.cpp_op import permute_tokens as cpp_move
from memmove.cuda_op import permute_tokens as cuda_move
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
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        results = []
        
        for i in range(self.runs):
            results.append(self.op(*args))
        
        end_event.record()
        
        torch.cuda.synchronize()
        end = time.time()
        cuda_elapse = start_event.elapsed_time(end_event) / self.runs * 1000
        
        data_size = results[0].element_size() * results[0].numel()
        
        elapse = (end - start) * (10 ** 6) / self.runs
        print(f"benchmark {self.name} takes: {elapse:.1f} us, cuda elpase: {cuda_elapse:.1f} us, bw : {8 * data_size / elapse / (10 ** 3):.1f} gbps")
        
        return results[0]
        
    def __call__(self, *args):
        return self.run(*args)
        

def get_args():
    parser = ArgumentParser()
    
    parser.add_argument("-d", "--dim", type=int, default=4096)
    parser.add_argument("-b", "--batch", type=int, default=64)
    parser.add_argument("-e", "--experts", type=int, default=None)
    parser.add_argument("-p", "--profile", type=str, default=None)
    parser.add_argument("-m", "--memory-track", action="store_true")
    
    args = parser.parse_args()
    return args

@torch.inference_mode
def main():
    args = get_args()
    
    if not args.experts:
        # generate mappings
        mappings = torch.randperm(args.batch, dtype=torch.int64, device="cuda:0")
    else:
        # generate expert id (bin index)
        pass
    
    if args.profile:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_stack=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name="torch_profile", worker_name=args.profile),
        )
        profiler.start()
    
    inputs = torch.randn((args.batch, args.dim), dtype=torch.float16, device="cuda:0")
    print(f"Input shape: {inputs.shape}, dtype={inputs.dtype}")
    
    if args.memory_track:
        torch.cuda.memory._record_memory_history()
    
    torchop = Benchmark("torch", torch_move)
    cudaop = Benchmark("cuda", cuda_move)
    tritonop = Benchmark("triton", triton_move)
    cppop = Benchmark("cpp", cpp_move)
    
    # NOTE: the torch op is a reverse operation of the other three
    torch_res = torchop(inputs, mappings)
    cuda_res = cudaop(inputs, mappings)
    triton_res = tritonop(inputs, mappings)
    cpp_res = cppop(inputs, mappings)
    
    
    cpu_tensor = torch.randn((args.batch, args.dim), dtype=torch.float16, device="cpu")
    gpu_tensor = torch.randn((args.batch, args.dim), dtype=torch.float16, device="cuda")
    
    D2H = Benchmark("D2H", gpu_to_cpu)
    H2D = Benchmark("H2D", cpu_to_gpu)
    
    D2H(gpu_tensor)
    H2D(cpu_tensor)
        
    if args.memory_track:
        torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")
    
    if args.profile:
        profiler.stop()
        

if __name__ == '__main__':
    main()