// test_math_ops.cpp
// Tests every math_ops function with a comprehensive set of inputs.
// Build against ORIGINAL or NEW library; outputs are compared externally.

#include "hardware/math_ops.h"
#include "hardware/arx_can.h"
#include <cstdio>
#include <cmath>
#include <cstdint>

static int g_pass = 0, g_fail = 0;

static void check(const char *name, double got, double expected, double tol = 1e-5)
{
    double err = std::fabs(got - expected);
    if (err <= tol) {
        printf("PASS %-40s got=%.8f\n", name, got);
        g_pass++;
    } else {
        printf("FAIL %-40s got=%.8f  expected=%.8f  err=%.2e\n",
               name, got, expected, err);
        g_fail++;
    }
}

// ---- fmaxf3 / fminf3 -------------------------------------------------------
static void test_fmaxf3()
{
    struct { float a, b, c, expected; } cases[] = {
        {1, 2, 3, 3}, {3, 2, 1, 3}, {2, 3, 1, 3},
        {-1,-2,-3,-1},{0,0,0,0},{5,5,3,5},
        {-10, 0, 10, 10}, {7.5f, 3.2f, 7.5f, 7.5f},
    };
    for (auto &c : cases) {
        char buf[64]; snprintf(buf,sizeof(buf),"fmaxf3(%.1f,%.1f,%.1f)",c.a,c.b,c.c);
        check(buf, fmaxf3(c.a,c.b,c.c), c.expected);
    }
}

static void test_fminf3()
{
    struct { float a, b, c, expected; } cases[] = {
        {1, 2, 3, 1}, {3, 2, 1, 1}, {2, 3, 1, 1},
        {-1,-2,-3,-3},{0,0,0,0},{5,5,8,5},
        {-10, 0, 10,-10}, {7.5f, 3.2f, 7.5f, 3.2f},
    };
    for (auto &c : cases) {
        char buf[64]; snprintf(buf,sizeof(buf),"fminf3(%.1f,%.1f,%.1f)",c.a,c.b,c.c);
        check(buf, fminf3(c.a,c.b,c.c), c.expected);
    }
}

// ---- limit_norm ------------------------------------------------------------
static void test_limit_norm()
{
    struct { float x, y, lim, ex, ey; } cases[] = {
        // already within limit → unchanged
        {1.0f, 0.0f, 2.0f,  1.0f, 0.0f},
        {0.0f, 1.0f, 2.0f,  0.0f, 1.0f},
        // exactly on limit → unchanged
        {3.0f, 4.0f, 5.0f,  3.0f, 4.0f},
        // exceeds limit → scale down
        {6.0f, 8.0f, 5.0f,  3.0f, 4.0f},
        {-6.0f, 8.0f, 5.0f, -3.0f, 4.0f},
        {0.0f, 10.0f, 3.0f, 0.0f, 3.0f},
    };
    for (auto &c : cases) {
        float x = c.x, y = c.y;
        limit_norm(&x, &y, c.lim);
        char bx[64], by[64];
        snprintf(bx,sizeof(bx),"limit_norm x (%.1f,%.1f,%.1f)",c.x,c.y,c.lim);
        snprintf(by,sizeof(by),"limit_norm y (%.1f,%.1f,%.1f)",c.x,c.y,c.lim);
        check(bx, x, c.ex);
        check(by, y, c.ey);
    }
}

// ---- float_to_uint / uint_to_float roundtrip --------------------------------
static void test_float_uint_roundtrip()
{
    struct { float val, xmin, xmax; int bits; } cases[] = {
        { 0.0f,  -12.5f, 12.5f, 16},
        { 6.25f, -12.5f, 12.5f, 16},
        {-12.5f, -12.5f, 12.5f, 16},
        { 12.5f, -12.5f, 12.5f, 16},
        { 0.0f,  -45.0f, 45.0f, 12},
        {22.5f,  -45.0f, 45.0f, 12},
        { 0.0f,    0.0f,500.0f, 12},
        {250.0f,   0.0f,500.0f, 12},
        { 2.5f,    0.0f,  5.0f,  9},
        { 0.0f,  -18.0f, 18.0f, 12},
        { 9.0f,  -18.0f, 18.0f, 12},
    };
    // quantisation error = (xmax-xmin)/(2^bits - 1)
    for (auto &c : cases) {
        int   u = float_to_uint(c.val, c.xmin, c.xmax, c.bits);
        float v = uint_to_float(u,     c.xmin, c.xmax, c.bits);
        float tol = (c.xmax - c.xmin) / (float)((1 << c.bits) - 1) + 1e-5f;
        char buf[80];
        snprintf(buf,sizeof(buf),"f2u→u2f(%.4f,[%.1f,%.1f],%dbit)",
                 c.val,c.xmin,c.xmax,c.bits);
        check(buf, v, c.val, tol);

        // also print the raw integer so we can diff it
        printf("      uint_val=%d\n", u);
    }
}

// ---- int_to_float ----------------------------------------------------------
static void test_int_to_float()
{
    // int_to_float(val, max_, min_, bits) → min_ + val/65535*(max_-min_)
    struct { int val; float max_, min_; int bits; float expected; } cases[] = {
        {0,      100.0f,  0.0f, 16,  0.0f},
        {65535,  100.0f,  0.0f, 16,  100.0f},
        {32767,  100.0f,  0.0f, 16,  49.9992f},
        {0,       12.5f,-12.5f, 16, -12.5f},
        {65535,   12.5f,-12.5f, 16,  12.5f},
    };
    for (auto &c : cases) {
        char buf[80];
        snprintf(buf,sizeof(buf),"int_to_float(%d,[%.1f,%.1f])",c.val,c.min_,c.max_);
        check(buf, int_to_float(c.val,c.max_,c.min_,c.bits), c.expected, 5e-4);
    }
}

// ---- float16 / float32 roundtrip -------------------------------------------
static void test_float16_roundtrip()
{
    float inputs[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
                      100.0f, -100.0f, 0.001f, 3.14159f};
    for (float f32_in : inputs) {
        unsigned short f16;
        float f32_out;
        float32_to_float16(&f32_in, &f16);
        float16_to_float32(&f16, &f32_out);

        // max relative error for float16 is ~0.1%
        float tol = std::fabs(f32_in) * 1e-3f + 1e-4f;
        char buf[64];
        snprintf(buf,sizeof(buf),"f16roundtrip(%.5f) f16=0x%04x",f32_in,(unsigned)f16);
        check(buf, f32_out, f32_in, tol);
    }
}

// ---- DM motor CAN frame encoding -------------------------------------------
// Verify exact byte layout of send_DM_motor_cmd encoding:
//   pos(16bit) | vel(12bit) | kp(12bit) | kd(12bit) | tor(12bit) = 64 bits
static void test_dm_encoding()
{
    // known values
    float pos = 0.0f, vel = 0.0f, kp = 0.0f, kd = 0.0f, tor = 0.0f;
    int pi = float_to_uint(pos, -12.5f, 12.5f, 16);
    int vi = float_to_uint(vel, -45.0f, 45.0f, 12);
    int ki = float_to_uint(kp,    0.0f,500.0f, 12);
    int di = float_to_uint(kd,    0.0f,  5.0f, 12);
    int ti = float_to_uint(tor, -18.0f, 18.0f, 12);

    uint8_t data[8];
    data[0] = (uint8_t)(pi >> 8);
    data[1] = (uint8_t)(pi & 0xFF);
    data[2] = (uint8_t)(vi >> 4);
    data[3] = (uint8_t)(((vi & 0xF) << 4) | (ki >> 8));
    data[4] = (uint8_t)(ki & 0xFF);
    data[5] = (uint8_t)(di >> 4);
    data[6] = (uint8_t)(((di & 0xF) << 4) | (ti >> 8));
    data[7] = (uint8_t)(ti & 0xFF);

    printf("DM_encoding(0,0,0,0,0): ");
    for (int i = 0; i < 8; i++) printf("%02X ", data[i]);
    printf("\n");

    // mid-range test
    pos=3.14f; vel=10.0f; kp=100.0f; kd=2.0f; tor=5.0f;
    pi = float_to_uint(pos, -12.5f, 12.5f, 16);
    vi = float_to_uint(vel, -45.0f, 45.0f, 12);
    ki = float_to_uint(kp,    0.0f,500.0f, 12);
    di = float_to_uint(kd,    0.0f,  5.0f, 12);
    ti = float_to_uint(tor, -18.0f, 18.0f, 12);
    data[0]=(uint8_t)(pi>>8); data[1]=(uint8_t)(pi&0xFF);
    data[2]=(uint8_t)(vi>>4);
    data[3]=(uint8_t)(((vi&0xF)<<4)|(ki>>8));
    data[4]=(uint8_t)(ki&0xFF);
    data[5]=(uint8_t)(di>>4);
    data[6]=(uint8_t)(((di&0xF)<<4)|(ti>>8));
    data[7]=(uint8_t)(ti&0xFF);
    printf("DM_encoding(3.14,10,100,2,5): ");
    for (int i = 0; i < 8; i++) printf("%02X ", data[i]);
    printf("\n");
}

// ---- RV_can_data_repack / DM_can_data_repack --------------------------------
static void test_can_repack()
{
    // Build a DM-style feedback frame and unpack it
    std::array<OD_Motor_Msg, 10> msg{};

    // motor_id=2, pos=3.0, vel=5.0, tor=2.0
    float pos_ref = 3.0f, vel_ref = 5.0f, tor_ref = 2.0f;
    uint16_t pi = (uint16_t)float_to_uint(pos_ref, -12.5f, 12.5f, 16);
    uint16_t vi = (uint16_t)float_to_uint(vel_ref, -45.0f, 45.0f, 12);
    uint16_t ti = (uint16_t)float_to_uint(tor_ref, -18.0f, 18.0f, 12);

    uint8_t data[8] = {0};
    data[0] = 0x02;                                    // motor_id=2 in low nibble
    data[1] = (uint8_t)(pi >> 8);
    data[2] = (uint8_t)(pi & 0xFF);
    data[3] = (uint8_t)(vi >> 4);
    data[4] = (uint8_t)(((vi & 0xF) << 4) | (ti >> 8));
    data[5] = (uint8_t)(ti & 0xFF);

    DM_can_data_repack(data, msg);

    float tol_pos = (12.5f - (-12.5f)) / (float)((1<<16)-1) + 1e-5f;
    float tol_vel = (45.0f - (-45.0f)) / (float)((1<<12)-1) + 1e-5f;
    float tol_tor = (18.0f - (-18.0f)) / (float)((1<<12)-1) + 1e-5f;

    check("DM_repack motor_id",       msg[2].motor_id,            2,       0);
    check("DM_repack angle_rad",      msg[2].angle_actual_rad,    pos_ref, tol_pos);
    check("DM_repack speed_rad",      msg[2].speed_actual_rad,    vel_ref, tol_vel);
    check("DM_repack current_float",  msg[2].current_actual_float,tor_ref, tol_tor);

    // RV MIT-mode frame
    std::array<OD_Motor_Msg, 10> msg2{};
    float pos2=1.5f, vel2=-3.0f, tor2=0.5f;
    uint16_t pi2=(uint16_t)float_to_uint(pos2,-12.5f,12.5f,16);
    uint16_t vi2=(uint16_t)float_to_uint(vel2,-18.0f,18.0f,12);
    uint16_t ti2=(uint16_t)float_to_uint(tor2,-30.0f,30.0f,12);
    uint8_t d2[8]={0};
    d2[0]= 0x20;               // msg_type=1 (bits[7:5]=001)
    d2[1]=(uint8_t)(pi2>>8); d2[2]=(uint8_t)(pi2&0xFF);
    d2[3]=(uint8_t)(vi2>>4);
    d2[4]=(uint8_t)(((vi2&0xF)<<4)|(ti2>>8));
    d2[5]=(uint8_t)(ti2&0xFF);
    d2[6]=0x64; // temp=(100-50)/2=25°C

    RV_can_data_repack(0x00000003u, d2, 8, 0, msg2); // motor_id=3
    float tol_pos2=(12.5f-(-12.5f))/(float)((1<<16)-1)+1e-5f;
    float tol_vel2=(18.0f-(-18.0f))/(float)((1<<12)-1)+1e-5f;
    float tol_tor2=(30.0f-(-30.0f))/(float)((1<<12)-1)+1e-5f;
    check("RV_repack motor_id",      (double)msg2[3].motor_id,          3,       0);
    check("RV_repack angle_rad",     msg2[3].angle_actual_rad,    pos2,  tol_pos2);
    check("RV_repack speed_rad",     msg2[3].speed_actual_rad,    vel2,  tol_vel2);
    check("RV_repack current_float", msg2[3].current_actual_float,tor2,  tol_tor2);
}

int main()
{
    printf("=== math_ops tests ===\n");
    test_fmaxf3();
    test_fminf3();
    test_limit_norm();
    test_float_uint_roundtrip();
    test_int_to_float();
    test_float16_roundtrip();
    test_dm_encoding();
    test_can_repack();

    printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
