import warp as wp

wp.init()
print("WARP_CUDA_DEVICES:", wp.get_cuda_devices())
print("WARP_CUDA_AVAILABLE:", wp.is_cuda_available())
