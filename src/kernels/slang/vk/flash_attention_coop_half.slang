// F16 QK and PV operands with F32 softmax and accumulation.
#define VELRI_HALF_ATTENTION_OPERANDS 1
#define VELRI_PREFILL_KERNEL_ENTRY vk_prefill_coop_half_f32
#include "flash_attention_coop_common.slang"
