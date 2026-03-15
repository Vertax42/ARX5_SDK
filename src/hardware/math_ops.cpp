#include "hardware/math_ops.h"
#include <math.h>
#include <stdint.h>

float fmaxf3(float x, float y, float z)
{
    if (x <= y) {
        return (y <= z) ? z : y;
    }
    return (x <= z) ? z : x;
}

float fminf3(float x, float y, float z)
{
    if (y <= x) {
        return (z <= y) ? z : y;
    }
    return (z <= x) ? z : x;
}

void limit_norm(float *x, float *y, float limit)
{
    float norm = sqrtf((*x) * (*x) + (*y) * (*y));
    if (norm > limit) {
        *x = (*x * limit) / norm;
        *y = (*y * limit) / norm;
    }
}

int float_to_uint(float x, float x_min, float x_max, int bits)
{
    float span   = x_max - x_min;
    float offset = x - x_min;
    return (int)((float)((1 << bits) - 1) * offset / span);
}

float uint_to_float(int x_int, float x_min, float x_max, int bits)
{
    float span = x_max - x_min;
    return (float)x_int * span / (float)((1 << bits) - 1) + x_min;
}

float int_to_float(int val, float max_, float min_, int bits)
{
    (void)bits;
    return min_ + ((float)val / 65535.0f) * (max_ - min_);
}

// -------------------------------------------------------------------------
// Float16 (half precision) <-> Float32 conversion
// Standard IEEE 754-2008 half precision format:
//   1 sign bit | 5 exponent bits (bias 15) | 10 mantissa bits
// -------------------------------------------------------------------------
void float32_to_float16(float *float32, unsigned short int *float16)
{
    uint32_t f32;
    __builtin_memcpy(&f32, float32, sizeof(f32));

    uint32_t sign     = (f32 >> 16) & 0x8000u;
    int32_t  exp      = (int32_t)((f32 >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = (f32 >> 13) & 0x3FFu;

    // Direct bit packing (no underflow/overflow guard — matches original library)
    *float16 = (unsigned short)(sign | ((uint32_t)(exp & 0x1F) << 10) | mantissa);
}

void float16_to_float32(unsigned short int *float16, float *float32)
{
    uint16_t f16     = *float16;
    uint32_t sign    = (uint32_t)(f16 & 0x8000u) << 16;
    int32_t  exp_h   = (f16 >> 10) & 0x1Fu;
    uint32_t mantissa = (uint32_t)(f16 & 0x3FFu) << 13;

    uint32_t f32;
    if (exp_h == 0) {
        f32 = sign; // ±0 (denormals treated as zero)
    } else if (exp_h == 31) {
        f32 = sign | 0x7F800000u | mantissa; // ±inf or NaN
    } else {
        uint32_t exp_f = (uint32_t)(exp_h - 15 + 127);
        f32 = sign | (exp_f << 23) | mantissa;
    }
    __builtin_memcpy(float32, &f32, sizeof(f32));
}
