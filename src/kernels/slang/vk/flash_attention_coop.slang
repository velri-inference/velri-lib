// Dense F16 KV prefill with two half operands representing each F32 query and probability.
#define VELRI_PREFILL_KERNEL_ENTRY vk_prefill_coop_f32
#include "flash_attention_coop_common.slang"
