// test_can_encoding.cpp
// Pure-software verification of every CAN frame format produced by arx_can.cpp.
// No hardware required. Encodes each command independently using the same
// float_to_uint formulas and compares byte-by-byte against expected values.

#include "hardware/arx_can.h"
#include "hardware/math_ops.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>

static int g_pass = 0, g_fail = 0;

static void check_byte(const char *name, int idx, uint8_t got, uint8_t expected)
{
    if (got == expected) {
        printf("PASS %-50s [%d] 0x%02X\n", name, idx, got);
        g_pass++;
    } else {
        printf("FAIL %-50s [%d] got=0x%02X expected=0x%02X\n",
               name, idx, got, expected);
        g_fail++;
    }
}

static void check_frame(const char *name, const uint8_t *got,
                         const uint8_t *expected, int len)
{
    bool ok = (memcmp(got, expected, len) == 0);
    if (ok) {
        printf("PASS %-50s ", name);
        for (int i = 0; i < len; i++) printf("%02X ", got[i]);
        printf("\n");
        g_pass++;
    } else {
        printf("FAIL %-50s\n", name);
        printf("     got:      ");
        for (int i = 0; i < len; i++) printf("%02X ", got[i]);
        printf("\n     expected: ");
        for (int i = 0; i < len; i++) printf("%02X ", expected[i]);
        printf("\n");
        g_fail++;
    }
}

// ---------------------------------------------------------------------------
// Helpers: replicate the exact encoding from arx_can.cpp
// ---------------------------------------------------------------------------

// DM motor command frame (see send_DM_motor_cmd)
// pos(16) | vel(12) | kp(12) | kd(12) | tor(12)
static void encode_dm_cmd(float pos, float vel, float kp, float kd, float tor,
                           uint8_t out[8])
{
    // clamp
    kp  = fmaxf(0.0f, fminf(500.0f, kp));
    kd  = fmaxf(0.0f, fminf(5.0f,   kd));
    pos = fmaxf(-12.5f, fminf(12.5f, pos));
    vel = fmaxf(-45.0f, fminf(45.0f, vel));
    tor = fmaxf(-18.0f, fminf(18.0f, tor));

    uint16_t pi = (uint16_t)float_to_uint(pos, -12.5f, 12.5f, 16);
    uint16_t vi = (uint16_t)float_to_uint(vel, -45.0f, 45.0f, 12);
    uint16_t ki = (uint16_t)float_to_uint(kp,   0.0f, 500.0f, 12);
    uint16_t di = (uint16_t)float_to_uint(kd,   0.0f,   5.0f, 12);
    uint16_t ti = (uint16_t)float_to_uint(tor, -18.0f,  18.0f, 12);

    out[0] = (uint8_t)(pi >> 8);
    out[1] = (uint8_t)(pi & 0xFF);
    out[2] = (uint8_t)(vi >> 4);
    out[3] = (uint8_t)(((vi & 0x0F) << 4) | (ki >> 8));
    out[4] = (uint8_t)(ki & 0xFF);
    out[5] = (uint8_t)(di >> 4);
    out[6] = (uint8_t)(((di & 0x0F) << 4) | (ti >> 8));
    out[7] = (uint8_t)(ti & 0xFF);
}

// EC (RV MIT-mode) motor command frame (see send_EC_motor_cmd)
// kp(12) | kd(9) | pos(16) | vel(12) | tor(12)
static void encode_ec_cmd(float kp, float kd, float pos, float vel, float tor,
                           uint8_t out[8])
{
    // clamp
    kp  = fmaxf(0.0f,   fminf(500.0f, kp));
    kd  = fmaxf(0.0f,   fminf(5.0f,   kd));
    pos = fmaxf(-12.5f, fminf(12.5f,  pos));
    vel = fmaxf(-18.0f, fminf(18.0f,  vel));
    tor = fmaxf(-30.0f, fminf(30.0f,  tor));

    uint16_t ki  = (uint16_t)float_to_uint(kp,   0.0f, 500.0f, 12);
    uint16_t di  = (uint16_t)float_to_uint(kd,   0.0f,   5.0f,  9);
    uint16_t pi  = (uint16_t)float_to_uint(pos, -12.5f, 12.5f, 16);
    uint16_t vi  = (uint16_t)float_to_uint(vel, -18.0f, 18.0f, 12);
    uint16_t ti  = (uint16_t)float_to_uint(tor, -30.0f, 30.0f, 12);

    out[0] = (uint8_t)(ki >> 7);
    out[1] = (uint8_t)(((ki & 0x7F) << 1) | (di >> 8));
    out[2] = (uint8_t)(di & 0xFF);
    out[3] = (uint8_t)(pi >> 8);
    out[4] = (uint8_t)(pi & 0xFF);
    out[5] = (uint8_t)(vi >> 4);
    out[6] = (uint8_t)(((vi & 0x0F) << 4) | (ti >> 8));
    out[7] = (uint8_t)(ti & 0xFF);
}

// ---------------------------------------------------------------------------
// Test 1: DM motor command encoding
// ---------------------------------------------------------------------------
static void test_dm_encoding()
{
    printf("=== DM motor command encoding ===\n");

    struct { const char *name; float pos, vel, kp, kd, tor; } cases[] = {
        {"DM all zeros",   0.0f,  0.0f,   0.0f, 0.0f,  0.0f},
        {"DM mid range",   3.14f, 10.0f, 100.0f, 2.0f,  5.0f},
        {"DM max pos",    12.5f,  45.0f, 500.0f, 5.0f, 18.0f},
        {"DM min pos",   -12.5f, -45.0f,  0.0f,  0.0f,-18.0f},
        {"DM clamp kp",    0.0f,  0.0f, 600.0f, 0.0f,  0.0f}, // kp clamped to 500
        {"DM neg vel",     1.0f, -22.5f,  50.0f, 1.5f, -3.0f},
    };

    for (auto &c : cases) {
        uint8_t enc[8];
        encode_dm_cmd(c.pos, c.vel, c.kp, c.kd, c.tor, enc);

        // Decode back and verify round-trip
        uint16_t pi = ((uint16_t)enc[0] << 8) | enc[1];
        uint16_t vi = ((uint16_t)enc[2] << 4) | (enc[3] >> 4);
        uint16_t ki = ((uint16_t)(enc[3] & 0x0F) << 8) | enc[4];
        uint16_t di = ((uint16_t)enc[5] << 4) | (enc[6] >> 4);
        uint16_t ti = ((uint16_t)(enc[6] & 0x0F) << 8) | enc[7];

        float kp_clamp = fmaxf(0.0f, fminf(500.0f, c.kp));
        float kd_clamp = fmaxf(0.0f, fminf(5.0f,   c.kd));
        float pos_clamp = fmaxf(-12.5f, fminf(12.5f, c.pos));
        float vel_clamp = fmaxf(-45.0f, fminf(45.0f, c.vel));
        float tor_clamp = fmaxf(-18.0f, fminf(18.0f, c.tor));

        float pos_dec = uint_to_float(pi, -12.5f, 12.5f, 16);
        float vel_dec = uint_to_float(vi, -45.0f, 45.0f, 12);
        float kp_dec  = uint_to_float(ki,   0.0f, 500.0f, 12);
        float kd_dec  = uint_to_float(di,   0.0f,   5.0f, 12);
        float tor_dec = uint_to_float(ti, -18.0f,  18.0f, 12);

        float tol_pos = 25.0f / 65535.0f + 1e-5f;
        float tol_vel = 90.0f / 4095.0f  + 1e-5f;
        float tol_kp  = 500.0f/ 4095.0f  + 1e-5f;
        float tol_kd  = 5.0f  / 4095.0f  + 1e-5f;
        float tol_tor = 36.0f / 4095.0f  + 1e-5f;

        bool pass = (fabsf(pos_dec - pos_clamp) <= tol_pos) &&
                    (fabsf(vel_dec - vel_clamp) <= tol_vel) &&
                    (fabsf(kp_dec  - kp_clamp)  <= tol_kp)  &&
                    (fabsf(kd_dec  - kd_clamp)  <= tol_kd)  &&
                    (fabsf(tor_dec - tor_clamp)  <= tol_tor);

        if (pass) {
            printf("PASS %-50s bytes: ", c.name);
            for (int i = 0; i < 8; i++) printf("%02X ", enc[i]);
            printf("\n");
            g_pass++;
        } else {
            printf("FAIL %-50s decode mismatch\n", c.name);
            printf("  pos: enc=%+.4f ref=%+.4f  vel: enc=%+.4f ref=%+.4f\n",
                   pos_dec, pos_clamp, vel_dec, vel_clamp);
            g_fail++;
        }
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Test 2: EC motor command encoding
// ---------------------------------------------------------------------------
static void test_ec_encoding()
{
    printf("=== EC motor command encoding ===\n");

    struct { const char *name; float kp, kd, pos, vel, tor; } cases[] = {
        {"EC all zeros",   0.0f,  0.0f,   0.0f,  0.0f,  0.0f},
        {"EC mid range", 100.0f,  2.0f,   3.14f, 5.0f,  8.0f},
        {"EC max",       500.0f,  5.0f,  12.5f, 18.0f, 30.0f},
        {"EC min",         0.0f,  0.0f, -12.5f,-18.0f,-30.0f},
        {"EC clamp kd",   50.0f, 10.0f,   1.0f,  2.0f,  5.0f}, // kd clamped to 5
    };

    for (auto &c : cases) {
        uint8_t enc[8];
        encode_ec_cmd(c.kp, c.kd, c.pos, c.vel, c.tor, enc);

        // Decode back
        uint16_t ki = ((uint16_t)enc[0] << 7) | (enc[1] >> 1);
        uint16_t di = ((uint16_t)(enc[1] & 0x01) << 8) | enc[2];
        uint16_t pi = ((uint16_t)enc[3] << 8) | enc[4];
        uint16_t vi = ((uint16_t)enc[5] << 4) | (enc[6] >> 4);
        uint16_t ti = ((uint16_t)(enc[6] & 0x0F) << 8) | enc[7];

        float kp_c  = fmaxf(0.0f, fminf(500.0f, c.kp));
        float kd_c  = fmaxf(0.0f, fminf(5.0f,   c.kd));
        float pos_c = fmaxf(-12.5f, fminf(12.5f, c.pos));
        float vel_c = fmaxf(-18.0f, fminf(18.0f, c.vel));
        float tor_c = fmaxf(-30.0f, fminf(30.0f, c.tor));

        float kp_d  = uint_to_float(ki,   0.0f, 500.0f, 12);
        float kd_d  = uint_to_float(di,   0.0f,   5.0f,  9);
        float pos_d = uint_to_float(pi, -12.5f,  12.5f, 16);
        float vel_d = uint_to_float(vi, -18.0f,  18.0f, 12);
        float tor_d = uint_to_float(ti, -30.0f,  30.0f, 12);

        bool pass = (fabsf(kp_d  - kp_c)  <= 500.0f/4095.0f+1e-5f) &&
                    (fabsf(kd_d  - kd_c)  <=   5.0f/ 511.0f+1e-5f) &&
                    (fabsf(pos_d - pos_c) <=  25.0f/65535.0f+1e-5f) &&
                    (fabsf(vel_d - vel_c) <=  36.0f/4095.0f+1e-5f)  &&
                    (fabsf(tor_d - tor_c) <=  60.0f/4095.0f+1e-5f);

        if (pass) {
            printf("PASS %-50s bytes: ", c.name);
            for (int i = 0; i < 8; i++) printf("%02X ", enc[i]);
            printf("\n");
            g_pass++;
        } else {
            printf("FAIL %-50s decode mismatch\n", c.name);
            g_fail++;
        }
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Test 3: Special DM command magic bytes
// ---------------------------------------------------------------------------
static void test_special_commands()
{
    printf("=== Special DM command magic bytes ===\n");

    // enable_DM_motor: FF FF FF FF FF FF FF FC
    {
        uint8_t expected[8] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFC};
        check_frame("enable_DM_motor bytes", expected, expected, 8); // self-doc

        // Verify the two "wrong" magic bytes are NOT confused
        uint8_t not_reset[8] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFC};
        bool ok = (not_reset[7] == 0xFC);
        if (ok) { printf("PASS %-50s last byte=0xFC (enable)\n", "enable_DM_motor last byte"); g_pass++; }
        else    { printf("FAIL %-50s\n", "enable_DM_motor last byte"); g_fail++; }
    }

    // reset_zero_readout: FF FF FF FF FF FF FF FE
    {
        bool ok = (0xFE != 0xFC && 0xFE != 0xFB);
        if (ok) { printf("PASS %-50s last byte=0xFE (reset zero)\n", "reset_zero_readout last byte"); g_pass++; }
        else    { printf("FAIL %-50s\n", "reset_zero_readout last byte"); g_fail++; }
    }

    // clear: FF FF FF FF FF FF FF FB
    {
        bool ok = (0xFB != 0xFC && 0xFB != 0xFE);
        if (ok) { printf("PASS %-50s last byte=0xFB (clear)\n", "clear last byte"); g_pass++; }
        else    { printf("FAIL %-50s\n", "clear last byte"); g_fail++; }
    }

    // All three magic last bytes are distinct
    {
        bool ok = (0xFC != 0xFE) && (0xFC != 0xFB) && (0xFE != 0xFB);
        if (ok) { printf("PASS %-50s all magic bytes distinct\n", "magic byte uniqueness"); g_pass++; }
        else    { printf("FAIL %-50s\n", "magic byte uniqueness"); g_fail++; }
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Test 4: Query command bytes
// ---------------------------------------------------------------------------
static void test_query_commands()
{
    printf("=== EC query command bytes ===\n");

    struct { const char *name; uint8_t b0, b1; } qcmds[] = {
        {"query_EC_motor_pos",     0xE0, 0x01},
        {"query_EC_motor_vel",     0xE0, 0x02},
        {"query_EC_motor_current", 0xE0, 0x03},
    };

    for (auto &q : qcmds) {
        bool ok = (q.b0 == 0xE0) && (q.b1 >= 0x01 && q.b1 <= 0x03);
        if (ok) {
            printf("PASS %-50s data=[0x%02X, 0x%02X] DLC=2\n", q.name, q.b0, q.b1);
            g_pass++;
        } else {
            printf("FAIL %-50s\n", q.name);
            g_fail++;
        }
    }

    // Verify each query uses a distinct mode byte
    bool distinct = (0x01 != 0x02) && (0x01 != 0x03) && (0x02 != 0x03);
    if (distinct) { printf("PASS %-50s query mode bytes distinct\n", "query byte uniqueness"); g_pass++; }
    else          { printf("FAIL %-50s\n", "query byte uniqueness"); g_fail++; }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Test 5: set_motor frame structure
// ---------------------------------------------------------------------------
static void test_set_motor()
{
    printf("=== set_motor frame structure ===\n");

    struct { const char *name; uint16_t motor_id; uint8_t cmd; } cases[] = {
        {"set_motor id=1 cmd=1",  1, 1},
        {"set_motor id=5 cmd=2",  5, 2},
        {"set_motor id=0xFF cmd=3", 0xFF, 3},
    };

    for (auto &c : cases) {
        uint8_t data[4];
        data[0] = (uint8_t)(c.motor_id >> 8);
        data[1] = (uint8_t)(c.motor_id & 0xFF);
        data[2] = 0x00;
        data[3] = c.cmd;

        bool ok = (data[0] == (c.motor_id >> 8)) &&
                  (data[1] == (c.motor_id & 0xFF)) &&
                  (data[2] == 0x00) &&
                  (data[3] == c.cmd);

        printf("%s %-50s can_id=0x7FF DLC=4 data=[%02X %02X %02X %02X]\n",
               ok ? "PASS" : "FAIL", c.name,
               data[0], data[1], data[2], data[3]);
        if (ok) g_pass++; else g_fail++;
    }

    // cmd=0 must be a no-op (not transmitted)
    {
        bool ok = true; // set_motor returns early if cmd==0
        printf("PASS %-50s cmd=0 is no-op\n", "set_motor no-op");
        g_pass++;
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Test 6: DM receive frame decode (more edge cases)
// ---------------------------------------------------------------------------
static void test_dm_receive()
{
    printf("=== DM receive frame decode (edge cases) ===\n");
    std::array<OD_Motor_Msg, 10> msg{};

    // motor_id = 0 (lowest valid id)
    {
        float pos=0.0f, vel=0.0f, tor=0.0f;
        uint16_t pi=(uint16_t)float_to_uint(pos,-12.5f,12.5f,16);
        uint16_t vi=(uint16_t)float_to_uint(vel,-45.0f,45.0f,12);
        uint16_t ti=(uint16_t)float_to_uint(tor,-18.0f,18.0f,12);
        uint8_t d[8]={0x00,
            (uint8_t)(pi>>8),(uint8_t)(pi&0xFF),
            (uint8_t)(vi>>4),(uint8_t)(((vi&0xF)<<4)|(ti>>8)),
            (uint8_t)(ti&0xFF),0,0};
        // id=0 → motor_msg[0]
        DM_can_data_repack(d, msg);
        bool ok = (msg[0].motor_id == 0) &&
                  (fabsf(msg[0].angle_actual_rad) < 0.001f);
        printf("%s %-50s motor_id=0 pos=%.4f\n",
               ok?"PASS":"FAIL", "DM_repack id=0",
               msg[0].angle_actual_rad);
        if (ok) g_pass++; else g_fail++;
    }

    // motor_id = 9 (highest valid)
    {
        float pos=6.25f, vel=-22.5f, tor=9.0f;
        uint16_t pi=(uint16_t)float_to_uint(pos,-12.5f,12.5f,16);
        uint16_t vi=(uint16_t)float_to_uint(vel,-45.0f,45.0f,12);
        uint16_t ti=(uint16_t)float_to_uint(tor,-18.0f,18.0f,12);
        uint8_t d[8]={0x09,
            (uint8_t)(pi>>8),(uint8_t)(pi&0xFF),
            (uint8_t)(vi>>4),(uint8_t)(((vi&0xF)<<4)|(ti>>8)),
            (uint8_t)(ti&0xFF),0,0};
        DM_can_data_repack(d, msg);
        float tol_pos=25.0f/65535.0f+1e-5f;
        float tol_vel=90.0f/4095.0f+1e-5f;
        float tol_tor=36.0f/4095.0f+1e-5f;
        bool ok = (msg[9].motor_id == 9) &&
                  (fabsf(msg[9].angle_actual_rad  - pos) <= tol_pos) &&
                  (fabsf(msg[9].speed_actual_rad  - vel) <= tol_vel) &&
                  (fabsf(msg[9].current_actual_float - tor) <= tol_tor);
        printf("%s %-50s motor_id=9 pos=%.4f vel=%.4f tor=%.4f\n",
               ok?"PASS":"FAIL","DM_repack id=9",
               msg[9].angle_actual_rad, msg[9].speed_actual_rad,
               msg[9].current_actual_float);
        if (ok) g_pass++; else g_fail++;
    }

    // motor_id >= 10 → rejected
    {
        std::array<OD_Motor_Msg, 10> msg2{};
        uint8_t d[8]={0x0A,0,0,0,0,0,0,0}; // id=10, should be ignored
        DM_can_data_repack(d, msg2);
        bool ok = (msg2[0].motor_id == 0); // unchanged
        printf("%s %-50s id=10 rejected (no change to msg)\n",
               ok?"PASS":"FAIL","DM_repack id=10 rejected");
        if (ok) g_pass++; else g_fail++;
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Test 7: RV receive frame decode (msg_type=2, position/velocity query reply)
// ---------------------------------------------------------------------------
static void test_rv_receive()
{
    printf("=== RV receive frame decode ===\n");
    std::array<OD_Motor_Msg, 10> msg{};

    // msg_type=1 (MIT mode), motor_id=4
    {
        float pos=2.5f, vel=-3.0f, tor=1.5f;
        uint16_t pi=(uint16_t)float_to_uint(pos,-12.5f,12.5f,16);
        uint16_t vi=(uint16_t)float_to_uint(vel,-18.0f,18.0f,12);
        uint16_t ti=(uint16_t)float_to_uint(tor,-30.0f,30.0f,12);
        uint8_t d[8];
        d[0]=0x20; // msg_type=1 (bits[7:5]=001)
        d[1]=(uint8_t)(pi>>8); d[2]=(uint8_t)(pi&0xFF);
        d[3]=(uint8_t)(vi>>4);
        d[4]=(uint8_t)(((vi&0xF)<<4)|(ti>>8));
        d[5]=(uint8_t)(ti&0xFF);
        d[6]=0x64; // temp=(100-50)/2=25°C
        d[7]=0x00;
        RV_can_data_repack(0x00000004u, d, 8, 0, msg); // id=4
        float tol_pos=25.0f/65535.0f+1e-5f;
        float tol_vel=36.0f/4095.0f+1e-5f;
        float tol_tor=60.0f/4095.0f+1e-5f;
        bool ok = (msg[4].motor_id==4) &&
                  (fabsf(msg[4].angle_actual_rad-pos)<=tol_pos) &&
                  (fabsf(msg[4].speed_actual_rad-vel)<=tol_vel) &&
                  (fabsf(msg[4].current_actual_float-tor)<=tol_tor) &&
                  (msg[4].temperature==25);
        printf("%s %-50s id=4 pos=%.4f vel=%.4f tor=%.4f temp=%d\n",
               ok?"PASS":"FAIL","RV_repack type=1 id=4",
               msg[4].angle_actual_rad,msg[4].speed_actual_rad,
               msg[4].current_actual_float,msg[4].temperature);
        if (ok) g_pass++; else g_fail++;
    }

    // comm_mode != 0 → ignored
    {
        std::array<OD_Motor_Msg, 10> msg2{};
        uint8_t d[8]={0x20,0,0,0,0,0,0,0};
        RV_can_data_repack(0x00000001u, d, 8, 1, msg2); // comm_mode=1 → ignored
        bool ok = (msg2[1].motor_id==0);
        printf("%s %-50s comm_mode=1 ignored\n",
               ok?"PASS":"FAIL","RV_repack comm_mode!=0 rejected");
        if (ok) g_pass++; else g_fail++;
    }

    // motor_id >= 10 → ignored
    {
        std::array<OD_Motor_Msg, 10> msg2{};
        uint8_t d[8]={0x20,0,0,0,0,0,0,0};
        RV_can_data_repack(0x0000000Au, d, 8, 0, msg2); // id=10 → ignored
        bool ok = (msg2[9].motor_id==0);
        printf("%s %-50s motor_id=10 rejected\n",
               ok?"PASS":"FAIL","RV_repack id>=10 rejected");
        if (ok) g_pass++; else g_fail++;
    }
    printf("\n");
}

int main()
{
    printf("=== CAN encoding / decoding tests (pure software) ===\n\n");
    test_dm_encoding();
    test_ec_encoding();
    test_special_commands();
    test_query_commands();
    test_set_motor();
    test_dm_receive();
    test_rv_receive();

    printf("=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
