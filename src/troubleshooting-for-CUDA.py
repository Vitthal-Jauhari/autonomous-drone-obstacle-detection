# gpu_diagnostic.py
import torch
import subprocess
import sys

def check_gpu_availability():
    print("=" * 50)
    print("GPU DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Basic PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Check if CUDA drivers are installed
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        print("\nNVIDIA-SMI output:")
        print(nvidia_smi.decode('utf-8'))
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nNVIDIA-SMI not found. NVIDIA drivers may not be installed.")
    
    # Check CUDA paths
    print("\nCUDA paths:")
    cuda_path = torch._C._cuda_getCompiledVersion()
    print(f"Compiled with CUDA: {cuda_path}")
    
    # Check if PyTorch was built with CUDA support
    print(f"PyTorch built with CUDA: {torch.cuda.is_built()}")
    
    # Try to create a tensor on GPU
    if torch.cuda.is_available():
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"Successfully created tensor on GPU: {x.device}")
            print(f"Tensor operations work: {x + 1}")
        except Exception as e:
            print(f"Error creating tensor on GPU: {e}")
    else:
        print("CUDA not available for tensor operations")

if __name__ == "__main__":
    # check_gpu_availability()
    # Run this to see what's available
    import utils
    print(dir(utils))