from natten.functional import get_device_cc

cc = get_device_cc()
cc = get_device_cc(0) # Optionally select a specific GPU

print(f"Your device is SM{cc}.")


import natten

# Whether NATTEN was built with CUDA kernels, 
# and supports running them on this system.
print(natten.has_cuda())

# Whether NATTEN supports running float16 on
# the selected device.
print(natten.has_half())
print(natten.has_half(0)) # Optionally specify a GPU index.

# Whether NATTEN supports running bfloat16 on
# the selected device.
print(natten.has_bfloat())
print(natten.has_bfloat(0)) # Optionally specify a GPU index.

# Whether NATTEN supports running GEMM kernels
# on the selected device.
print(natten.has_gemm())
print(natten.has_gemm(0)) # Optionally specify a GPU index.

# Whether NATTEN supports running GEMM kernels
# in full precision on the selected device.
print(natten.has_fp32_gemm())
print(natten.has_fp32_gemm(0)) # Optionally specify a GPU index.