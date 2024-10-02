import torch

if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    cached_memory = torch.cuda.memory_reserved(0)

    print(f"Total Memory: {total_memory / (1024**3):.2f} GB")
    print(f"Allocated Memory: {allocated_memory / (1024**3):.2f} GB")
    print(f"Cached Memory: {cached_memory / (1024**3):.2f} GB")
else:
    print("CUDA is not available")