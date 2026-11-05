#pragma once

#include "common.hpp"

// [a770-opt] single-token GQA flash-decoding for an f16 KV cache on Arc Alchemist (see fattn-xe1dec.cpp)
bool ggml_sycl_flash_attn_ext_xe1dec_supported(int device, const ggml_tensor * dst);

// [a770-opt] Arc Alchemist (Xe-HPG: A-series, DG2): picks the A770-tuned defaults
bool ggml_sycl_device_is_alchemist(int device);
void ggml_sycl_flash_attn_ext_xe1dec(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
