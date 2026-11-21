// Unified C-ABI CPU entrypoints for Velri-lib Slang kernels.
// Compiled ONCE per target ISA via -DVELRI_ISA=base|avx2|avx512 and -llvm-features
// with ZERO duplicated source code between ISA variants.
import numeric;
import activations;
import epilogue;
import norm;
import rope;
import elementwise;
import linear_cpu;

#define VELRI_CAT2(a, b) a##b
#define VELRI_CAT(a, b) VELRI_CAT2(a, b)
#ifndef VELRI_ISA
#define VELRI_ISA base
#endif
#ifndef VELRI_MR
#define VELRI_MR 4
#endif
#ifndef VELRI_NR_F32
#define VELRI_NR_F32 8
#endif
#ifndef VELRI_NR_F64
#define VELRI_NR_F64 4
#endif
#define VSYM(name) VELRI_CAT(VELRI_CAT(velri_slang_, VELRI_ISA), VELRI_CAT(_, name))

EpilogueParams make_ep(Ptr<float, Access.Read> bias, Ptr<float, Access.Read> x_scale,
                       Ptr<float, Access.Read> w_scale, uint flags)
{
    EpilogueParams ep;
    ep.bias = bias;
    ep.x_scale = x_scale;
    ep.w_scale = w_scale;
    ep.has_bias = (flags & 1u);
    ep.has_x_scale = (flags & 2u);
    ep.has_w_scale = (flags & 4u);
    return ep;
}

// ---- Linear (GEMV + L1-Packed BLIS Outer-Product GEMM) ----

[DllExport] export __extern_cpp void VSYM(linear_f32)(
    Ptr<float, Access.Read> x, uint xs, Ptr<float, Access.Read> w, uint ws,
    Ptr<float, Access.Read> bias, uint flags, Ptr<float> y, uint ys,
    uint r0, uint r1, uint c0, uint c1, uint k)
{
    linear_slice_cpu<VELRI_MR, VELRI_NR_F32, ElemF32, ElemF32, ElemF32, EpBias>(
        x, xs, w, ws, y, ys, r0, r1, c0, c1, k, make_ep(bias, bias, bias, flags));
}

[DllExport] export __extern_cpp void VSYM(linear_f16_f32)(
    Ptr<ElemF16.Storage, Access.Read> x, uint xs, Ptr<ElemF16.Storage, Access.Read> w, uint ws,
    Ptr<float, Access.Read> bias, uint flags, Ptr<float> y, uint ys,
    uint r0, uint r1, uint c0, uint c1, uint k)
{
    linear_slice_cpu<VELRI_MR, VELRI_NR_F32, ElemF16, ElemF16, ElemF32, EpBias>(
        x, xs, w, ws, y, ys, r0, r1, c0, c1, k, make_ep(bias, bias, bias, flags));
}

[DllExport] export __extern_cpp void VSYM(linear_bf16_f32)(
    Ptr<uint16_t, Access.Read> x, uint xs, Ptr<uint16_t, Access.Read> w, uint ws,
    Ptr<float, Access.Read> bias, uint flags, Ptr<float> y, uint ys,
    uint r0, uint r1, uint c0, uint c1, uint k)
{
    linear_slice_cpu<VELRI_MR, VELRI_NR_F32, ElemBF16, ElemBF16, ElemF32, EpBias>(
        x, xs, w, ws, y, ys, r0, r1, c0, c1, k, make_ep(bias, bias, bias, flags));
}

[DllExport] export __extern_cpp void VSYM(linear_f32x_bf16w)(
    Ptr<float, Access.Read> x, uint xs, Ptr<uint16_t, Access.Read> w, uint ws,
    Ptr<float, Access.Read> bias, uint flags, Ptr<float> y, uint ys,
    uint r0, uint r1, uint c0, uint c1, uint k)
{
    linear_slice_cpu<VELRI_MR, VELRI_NR_F32, ElemF32, ElemBF16, ElemF32, EpBias>(
        x, xs, w, ws, y, ys, r0, r1, c0, c1, k, make_ep(bias, bias, bias, flags));
}

[DllExport] export __extern_cpp void VSYM(linear_f64)(
    Ptr<double, Access.Read> x, uint xs, Ptr<double, Access.Read> w, uint ws,
    Ptr<double, Access.Read> bias, uint has_bias, Ptr<double> y, uint ys,
    uint r0, uint r1, uint c0, uint c1, uint k)
{
    linear_f64_slice_cpu<VELRI_MR, VELRI_NR_F64>(x, xs, w, ws, bias, has_bias, y, ys, r0, r1, c0, c1, k);
}

[DllExport] export __extern_cpp void VSYM(linear_w8a8_f32)(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<float, Access.Read> bias, Ptr<float, Access.Read> x_scale,
    Ptr<float, Access.Read> w_scale, uint flags, Ptr<float> y, uint ys,
    uint r0, uint r1, uint c0, uint c1, uint k)
{
    linear_w8a8_f32_slice_cpu<VELRI_MR, VELRI_NR_F32>(
        x, xs, w, ws, y, ys, r0, r1, c0, c1, k, make_ep(bias, x_scale, w_scale, flags));
}

// ---- Normalization ----

[DllExport] export __extern_cpp void VSYM(rmsnorm_f32)(
    Ptr<float, Access.Read> x, uint xs, Ptr<float, Access.Read> w, uint has_w,
    Ptr<float> y, uint ys, uint r0, uint r1, uint dim, float eps, uint unit_offset)
{
    rmsnorm_rows_cpu<ElemF32, ElemF32>(x, xs, w, has_w, y, ys, r0, r1, dim, eps, unit_offset);
}

[DllExport] export __extern_cpp void VSYM(rmsnorm_bf16)(
    Ptr<uint16_t, Access.Read> x, uint xs, Ptr<uint16_t, Access.Read> w, uint has_w,
    Ptr<uint16_t> y, uint ys, uint r0, uint r1, uint dim, float eps, uint unit_offset)
{
    rmsnorm_rows_cpu<ElemBF16, ElemBF16>(x, xs, w, has_w, y, ys, r0, r1, dim, eps, unit_offset);
}

[DllExport] export __extern_cpp void VSYM(rmsnorm_f16)(
    Ptr<ElemF16.Storage, Access.Read> x, uint xs, Ptr<ElemF16.Storage, Access.Read> w, uint has_w,
    Ptr<ElemF16.Storage> y, uint ys, uint r0, uint r1, uint dim, float eps, uint unit_offset)
{
    rmsnorm_rows_cpu<ElemF16, ElemF16>(x, xs, w, has_w, y, ys, r0, r1, dim, eps, unit_offset);
}

[DllExport] export __extern_cpp void VSYM(fused_add_rmsnorm_f32)(
    Ptr<float> res, uint rs, Ptr<float, Access.Read> x, uint xs,
    Ptr<float, Access.Read> w, uint has_w, Ptr<float> y, uint ys,
    uint r0, uint r1, uint dim, float eps, uint unit_offset)
{
    fused_add_rmsnorm_rows_cpu(res, rs, x, xs, w, has_w, y, ys, r0, r1, dim, eps, unit_offset);
}

[DllExport] export __extern_cpp void VSYM(layernorm_f32)(
    Ptr<float, Access.Read> x, uint xs, Ptr<float, Access.Read> w, uint has_w,
    Ptr<float, Access.Read> b, uint has_b, Ptr<float> y, uint ys,
    uint r0, uint r1, uint dim, float eps)
{
    layernorm_rows_cpu(x, xs, w, has_w, b, has_b, y, ys, r0, r1, dim, eps);
}

// ---- RoPE ----

[DllExport] export __extern_cpp void VSYM(rope_f32)(
    Ptr<float, Access.Read> x, uint tsx, uint hsx,
    Ptr<float, Access.Read> cos_tbl, Ptr<float, Access.Read> sin_tbl,
    Ptr<float> y, uint tsy, uint hsy,
    uint t0, uint t1, uint n_heads, uint head_dim, uint rotary_dim,
    uint pos_offset, uint mode)
{
    rope_tokens_cpu<ElemF32>(x, tsx, hsx, cos_tbl, sin_tbl, y, tsy, hsy,
                             t0, t1, n_heads, head_dim, rotary_dim, pos_offset, mode);
}

[DllExport] export __extern_cpp void VSYM(rope_bf16)(
    Ptr<uint16_t, Access.Read> x, uint tsx, uint hsx,
    Ptr<float, Access.Read> cos_tbl, Ptr<float, Access.Read> sin_tbl,
    Ptr<uint16_t> y, uint tsy, uint hsy,
    uint t0, uint t1, uint n_heads, uint head_dim, uint rotary_dim,
    uint pos_offset, uint mode)
{
    rope_tokens_cpu<ElemBF16>(x, tsx, hsx, cos_tbl, sin_tbl, y, tsy, hsy,
                              t0, t1, n_heads, head_dim, rotary_dim, pos_offset, mode);
}

// ---- Activations, GatedMLP, Softmax, Vector Math, FWHT, Row Quantization ----

[DllExport] export __extern_cpp void VSYM(activation_silu_f32)(
    Ptr<float, Access.Read> x, Ptr<float> y, uint i0, uint i1)
{
    activation_cpu<ActSiLU>(x, y, i0, i1);
}

[DllExport] export __extern_cpp void VSYM(activation_gelu_f32)(
    Ptr<float, Access.Read> x, Ptr<float> y, uint i0, uint i1)
{
    activation_cpu<ActGeluTanh>(x, y, i0, i1);
}

[DllExport] export __extern_cpp void VSYM(activation_relu2_f32)(
    Ptr<float, Access.Read> x, Ptr<float> y, uint i0, uint i1)
{
    activation_cpu<ActReLU2>(x, y, i0, i1);
}

[DllExport] export __extern_cpp void VSYM(gated_silu_f32)(
    Ptr<float, Access.Read> gate, Ptr<float, Access.Read> up, Ptr<float> y,
    uint i0, uint i1, float clamp_limit)
{
    gated_mlp_cpu<ActSiLU>(gate, up, y, i0, i1, clamp_limit);
}

[DllExport] export __extern_cpp void VSYM(gated_gelu_f32)(
    Ptr<float, Access.Read> gate, Ptr<float, Access.Read> up, Ptr<float> y,
    uint i0, uint i1, float clamp_limit)
{
    gated_mlp_cpu<ActGeluTanh>(gate, up, y, i0, i1, clamp_limit);
}

[DllExport] export __extern_cpp void VSYM(softmax_f32)(
    Ptr<float, Access.Read> x, uint xs, Ptr<float> y, uint ys,
    uint r0, uint r1, uint cols)
{
    softmax_rows_cpu(x, xs, y, ys, r0, r1, cols);
}

[DllExport] export __extern_cpp void VSYM(vec_axpby_f32)(
    float alpha, Ptr<float, Access.Read> a, float beta, Ptr<float, Access.Read> b,
    Ptr<float> y, uint i0, uint i1)
{
    axpby_cpu(alpha, a, beta, b, y, i0, i1);
}

[DllExport] export __extern_cpp void VSYM(vec_mul_f32)(
    Ptr<float, Access.Read> a, Ptr<float, Access.Read> b, Ptr<float> y,
    uint i0, uint i1)
{
    mul_cpu(a, b, y, i0, i1);
}

[DllExport] export __extern_cpp void VSYM(fwht_f32)(
    Ptr<float, Access.Read> x, uint xs, Ptr<uint8_t, Access.Read> suh, uint suh_mode,
    Ptr<float> y, uint ys, uint r0, uint r1, uint cols, uint block_size)
{
    if (suh_mode == 1u) {
        fwht_rows_cpu<ElemF32, true>(x, xs, Ptr<float, Access.Read>(suh), y, ys, r0, r1, cols, block_size);
    } else if (suh_mode == 2u) {
        fwht_rows_cpu<ElemF16, true>(x, xs, Ptr<ElemF16.Storage, Access.Read>(suh), y, ys, r0, r1, cols, block_size);
    } else {
        fwht_rows_cpu<ElemF32, false>(x, xs, Ptr<float, Access.Read>(suh), y, ys, r0, r1, cols, block_size);
    }
}

[DllExport] export __extern_cpp void VSYM(quantize_row_f32_i8)(
    Ptr<float, Access.Read> x, uint xs, Ptr<int8_t> xi8, uint xi8s,
    Ptr<float> scale_out, uint r0, uint r1, uint cols)
{
    quantize_rows_f32_i8_cpu(x, xs, xi8, xi8s, scale_out, r0, r1, cols);
}
