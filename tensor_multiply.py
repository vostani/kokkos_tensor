import torch
import tenmul
import cupy as cp
import torch.utils.dlpack as dlpack

tenmul.init()

# Create a tensor on a device (e.g., CPU or CUDA)
tn = torch.randn(10, device='cuda')  # Example tensor
print("tensor", tn)

pointer = tn.data_ptr()
print(f"ptr: {pointer}, type: {type(pointer)}")

# Pass the data pointer of the tensor to the C++ extension
result = tenmul.process_tensor(pointer, pointer, tn.numel())

print("after process")

print(tn)

print(result)

# causes segfault
#print("result_float_ptr", result.float_ptr)

# wrapped in pycapsule
print("result_void_ptr", result.void_ptr)

# is the same address
print("ull_ptr", result.ull_ptr)

print("result_size", result.size)

print("tn[0]: ", tn[0])

# create cupy pointer to the memory
mem = cp.cuda.UnownedMemory(result.ull_ptr, result.size, owner=None)
memptr = cp.cuda.MemoryPointer(mem, offset=0)

# convert to pytorch tensor
arr = cp.ndarray(tn.shape, dtype=cp.float32, memptr=memptr)

# doesn't work as it complains about copy=False
# no buffer protocol or dlpack implemented
# https://github.com/pytorch/pytorch/blob/8461e7ed9e68d1b7274e69d5396ff343ac120568/torch/csrc/utils/tensor_new.cpp#L1789
#ntn = torch.asarray(arr, dtype=torch.float32, copy=False, device="cuda")

ntn = torch.as_tensor(arr, device="cuda")

print(ntn)
print("old tensor pointer: ", tn.data_ptr())
print("new tensor pointer: ", ntn.data_ptr())

tenmul.finalize()