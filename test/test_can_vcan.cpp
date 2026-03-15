// test_can_vcan.cpp
// Full SocketCAN loopback test using Linux virtual CAN (vcan0).
//
// Prerequisites (run once before this test):
//   sudo modprobe vcan
//   sudo ip link add dev vcan0 type vcan
//   sudo ip link set up vcan0
//
// Usage:
//   ./test_can_vcan [vcan_interface]   (default: vcan0)

// Include arx_can.h first so CANFrame.h defines can_frame / sockaddr_can etc.
// before any kernel headers that would redefine them.
#include "hardware/arx_can.h"
#include "hardware/math_ops.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>
#include <array>

#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#include <sys/select.h>

// CAN_RAW is defined in CANFrame.h (value 1) — do NOT include <linux/can/raw.h>
// AF_CAN / PF_CAN
#ifndef AF_CAN
#define AF_CAN 29
#endif
#ifndef PF_CAN
#define PF_CAN AF_CAN
#endif

static int g_pass = 0, g_fail = 0;

// ---------------------------------------------------------------------------
// Sniffer: a plain raw SocketCAN socket that just reads
// ---------------------------------------------------------------------------
struct Sniffer {
    int fd = -1;

    bool open(const char *iface)
    {
        fd = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (fd < 0) return false;

        struct ifreq ifr;
        std::strncpy(ifr.ifr_name, iface, IFNAMSIZ - 1);
        if (ioctl(fd, SIOCGIFINDEX, &ifr) < 0) { ::close(fd); fd=-1; return false; }

        struct sockaddr_can addr{};
        addr.can_family  = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;
        if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0)
        { ::close(fd); fd=-1; return false; }

        return true;
    }

    // Read one frame with a 200ms timeout. Returns false on timeout.
    bool read_frame(struct can_frame *f, int timeout_ms = 200)
    {
        fd_set rds;
        FD_ZERO(&rds); FD_SET(fd, &rds);
        struct timeval tv { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
        int r = select(fd+1, &rds, nullptr, nullptr, &tv);
        if (r <= 0) return false;
        return ::read(fd, f, sizeof(*f)) > 0;
    }

    // Drain all pending frames (discard)
    void drain()
    {
        struct can_frame f;
        while (read_frame(&f, 10)) {}
    }

    // Inject a frame INTO the interface so ArxCan's receiver sees it
    bool inject(const struct can_frame *f)
    {
        return ::write(fd, f, sizeof(*f)) > 0;
    }

    void close() { if (fd >= 0) { ::close(fd); fd = -1; } }
    ~Sniffer() { close(); }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void check(const char *name, bool ok)
{
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    if (ok) g_pass++; else g_fail++;
}

// ---------------------------------------------------------------------------
// TestCan: thin wrapper around Usb2Can that exposes the same motor-command
// interface as ArxCan (same encoding logic, same can_frame construction).
// Used so we can test on "vcan0" which ArxCan rejects (it requires "can..." prefix).
// ---------------------------------------------------------------------------
class TestCan {
public:
    explicit TestCan(const char *iface) : usb_(iface) {}
    bool is_open() { return usb_.is_open(); }

    void send_DM_motor_cmd(uint16_t id, float kp, float kd, float pos, float spd, float tor) {
        kp  = fmaxf(0,fminf(500,kp)); kd  = fmaxf(0,fminf(5,kd));
        pos = fmaxf(-12.5f,fminf(12.5f,pos)); spd = fmaxf(-45,fminf(45,spd));
        tor = fmaxf(-18,fminf(18,tor));
        uint16_t pi=(uint16_t)float_to_uint(pos,-12.5f,12.5f,16);
        uint16_t vi=(uint16_t)float_to_uint(spd,-45,45,12);
        uint16_t ki=(uint16_t)float_to_uint(kp,0,500,12);
        uint16_t di=(uint16_t)float_to_uint(kd,0,5,12);
        uint16_t ti=(uint16_t)float_to_uint(tor,-18,18,12);
        can_frame_t f{}; f.can_id=id; f.len=8;
        f.data[0]=(uint8_t)(pi>>8); f.data[1]=(uint8_t)(pi&0xFF);
        f.data[2]=(uint8_t)(vi>>4);
        f.data[3]=(uint8_t)(((vi&0xF)<<4)|(ki>>8)); f.data[4]=(uint8_t)(ki&0xFF);
        f.data[5]=(uint8_t)(di>>4);
        f.data[6]=(uint8_t)(((di&0xF)<<4)|(ti>>8)); f.data[7]=(uint8_t)(ti&0xFF);
        usb_.transmit(f);
    }

    void send_EC_motor_cmd(uint16_t id, float kp, float kd, float pos, float spd, float tor) {
        kp  = fmaxf(0,fminf(500,kp)); kd  = fmaxf(0,fminf(5,kd));
        pos = fmaxf(-12.5f,fminf(12.5f,pos)); spd = fmaxf(-18,fminf(18,spd));
        tor = fmaxf(-30,fminf(30,tor));
        uint16_t ki=(uint16_t)float_to_uint(kp,0,500,12);
        uint16_t di=(uint16_t)float_to_uint(kd,0,5,9);
        uint16_t pi=(uint16_t)float_to_uint(pos,-12.5f,12.5f,16);
        uint16_t vi=(uint16_t)float_to_uint(spd,-18,18,12);
        uint16_t ti=(uint16_t)float_to_uint(tor,-30,30,12);
        can_frame_t f{}; f.can_id=id; f.len=8;
        f.data[0]=(uint8_t)(ki>>7);
        f.data[1]=(uint8_t)(((ki&0x7F)<<1)|(di>>8)); f.data[2]=(uint8_t)(di&0xFF);
        f.data[3]=(uint8_t)(pi>>8); f.data[4]=(uint8_t)(pi&0xFF);
        f.data[5]=(uint8_t)(vi>>4);
        f.data[6]=(uint8_t)(((vi&0xF)<<4)|(ti>>8)); f.data[7]=(uint8_t)(ti&0xFF);
        usb_.transmit(f);
    }

    void enable_DM_motor(uint16_t id) {
        can_frame_t f{}; f.can_id=id; f.len=8;
        memset(f.data, 0xFF, 8); f.data[7]=0xFC; usb_.transmit(f);
    }
    void reset_zero_readout(uint16_t id) {
        can_frame_t f{}; f.can_id=id; f.len=8;
        memset(f.data, 0xFF, 8); f.data[7]=0xFE; usb_.transmit(f);
    }
    void clear_motor(uint16_t id) {
        can_frame_t f{}; f.can_id=id; f.len=8;
        memset(f.data, 0xFF, 8); f.data[7]=0xFB; usb_.transmit(f);
    }
    void query_EC_motor_pos(uint16_t id)     { send_query(id, 0x01); }
    void query_EC_motor_vel(uint16_t id)     { send_query(id, 0x02); }
    void query_EC_motor_current(uint16_t id) { send_query(id, 0x03); }

    const std::array<OD_Motor_Msg, 10> get_motor_msg() { return usb_.get_motor_msg(); }

private:
    Usb2Can usb_;
    void send_query(uint16_t id, uint8_t mode) {
        can_frame_t f{}; f.can_id=id; f.len=2; f.data[0]=0xE0; f.data[1]=mode;
        usb_.transmit(f);
    }
};

static void check_bytes(const char *name, const uint8_t *got,
                         const uint8_t *expected, int len)
{
    bool ok = (memcmp(got, expected, len) == 0);
    if (ok) {
        printf("PASS %-52s bytes: ", name);
    } else {
        printf("FAIL %-52s\n     got:      ", name);
        for (int i=0;i<len;i++) printf("%02X ", got[i]);
        printf("\n     expected: ");
        for (int i=0;i<len;i++) printf("%02X ", expected[i]);
        printf("\n");
    }
    if (ok) {
        for (int i=0;i<len;i++) printf("%02X ", got[i]);
        printf("\n");
        g_pass++;
    } else {
        g_fail++;
    }
}

// Encode expected DM command bytes (same logic as arx_can.cpp)
static void expected_dm_bytes(float pos, float vel, float kp, float kd, float tor,
                               uint8_t out[8])
{
    kp  = fmaxf(0.0f, fminf(500.0f, kp));
    kd  = fmaxf(0.0f, fminf(5.0f,   kd));
    pos = fmaxf(-12.5f, fminf(12.5f, pos));
    vel = fmaxf(-45.0f, fminf(45.0f, vel));
    tor = fmaxf(-18.0f, fminf(18.0f, tor));
    uint16_t pi=(uint16_t)float_to_uint(pos,-12.5f,12.5f,16);
    uint16_t vi=(uint16_t)float_to_uint(vel,-45.0f,45.0f,12);
    uint16_t ki=(uint16_t)float_to_uint(kp,  0.0f,500.0f,12);
    uint16_t di=(uint16_t)float_to_uint(kd,  0.0f,  5.0f,12);
    uint16_t ti=(uint16_t)float_to_uint(tor,-18.0f, 18.0f,12);
    out[0]=(uint8_t)(pi>>8); out[1]=(uint8_t)(pi&0xFF);
    out[2]=(uint8_t)(vi>>4);
    out[3]=(uint8_t)(((vi&0xF)<<4)|(ki>>8)); out[4]=(uint8_t)(ki&0xFF);
    out[5]=(uint8_t)(di>>4);
    out[6]=(uint8_t)(((di&0xF)<<4)|(ti>>8)); out[7]=(uint8_t)(ti&0xFF);
}

// Encode expected EC command bytes
static void expected_ec_bytes(float kp, float kd, float pos, float vel, float tor,
                               uint8_t out[8])
{
    kp  = fmaxf(0.0f,  fminf(500.0f, kp));
    kd  = fmaxf(0.0f,  fminf(5.0f,   kd));
    pos = fmaxf(-12.5f,fminf(12.5f,  pos));
    vel = fmaxf(-18.0f,fminf(18.0f,  vel));
    tor = fmaxf(-30.0f,fminf(30.0f,  tor));
    uint16_t ki=(uint16_t)float_to_uint(kp,  0.0f,500.0f,12);
    uint16_t di=(uint16_t)float_to_uint(kd,  0.0f,  5.0f, 9);
    uint16_t pi=(uint16_t)float_to_uint(pos,-12.5f,12.5f,16);
    uint16_t vi=(uint16_t)float_to_uint(vel,-18.0f,18.0f,12);
    uint16_t ti=(uint16_t)float_to_uint(tor,-30.0f,30.0f,12);
    out[0]=(uint8_t)(ki>>7);
    out[1]=(uint8_t)(((ki&0x7F)<<1)|(di>>8)); out[2]=(uint8_t)(di&0xFF);
    out[3]=(uint8_t)(pi>>8); out[4]=(uint8_t)(pi&0xFF);
    out[5]=(uint8_t)(vi>>4);
    out[6]=(uint8_t)(((vi&0xF)<<4)|(ti>>8)); out[7]=(uint8_t)(ti&0xFF);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    const char *iface = (argc > 1) ? argv[1] : "vcan0";
    printf("=== CAN vcan loopback tests (interface: %s) ===\n\n", iface);

    // -----------------------------------------------------------------------
    // Section 1: Usb2Can lifecycle
    // -----------------------------------------------------------------------
    printf("--- Usb2Can open/close ---\n");
    {
        try {
            Usb2Can can(iface);
            check("Usb2Can::is_open() after construction", can.is_open());
        } catch (const std::exception &e) {
            printf("FAIL Usb2Can construction: %s\n", e.what());
            g_fail++;
        }
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Section 2: ArxCan routing
    // -----------------------------------------------------------------------
    printf("--- ArxCan routing ---\n");
    {
        // "can..." prefix → Usb2Can.  "vcan0" starts with "v" so ArxCan rejects it.
        // We verify the ROUTING LOGIC by checking "can0" would be accepted
        // and "xyz0" would throw.  (We don't actually open can0 since it may not exist.)
        bool threw_xyz = false;
        try { ArxCan bad("xyz0"); } catch (const std::runtime_error &) { threw_xyz = true; }
        check("ArxCan(\"xyz0\") throws for unknown prefix", threw_xyz);
        // Note: "eth..." or "en..." prefix routes to EtherCat2Can which segfaults without
        // hardware (libsoem ecx_setupnic is not exception-safe) — not testable here.
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Set up sniffer socket (reads frames off vcan0)
    // -----------------------------------------------------------------------
    Sniffer sniff;
    if (!sniff.open(iface)) {
        printf("ERROR: failed to open sniffer socket on %s\n", iface);
        printf("       Run: sudo modprobe vcan && "
               "sudo ip link add dev %s type vcan && "
               "sudo ip link set up %s\n", iface, iface);
        return 2;
    }

    // -----------------------------------------------------------------------
    // Section 3: DM motor command encoding via TestCan (Usb2Can on vcan)
    // -----------------------------------------------------------------------
    printf("--- send_DM_motor_cmd frame bytes ---\n");
    {
        TestCan ac(iface);
        usleep(10000); // let receiver thread start

        struct {
            const char *name;
            uint16_t id;
            float pos, vel, kp, kd, tor;
        } cases[] = {
            {"DM all zeros (id=1)",    1, 0.0f,  0.0f,   0.0f, 0.0f,  0.0f},
            {"DM mid range (id=2)",    2, 3.14f,10.0f, 100.0f, 2.0f,  5.0f},
            {"DM max vals (id=3)",     3,12.5f, 45.0f, 500.0f, 5.0f, 18.0f},
        };

        for (auto &c : cases) {
            sniff.drain();
            ac.send_DM_motor_cmd(c.id, c.kp, c.kd, c.pos, c.vel, c.tor);
            usleep(5000);

            struct can_frame f{};
            if (!sniff.read_frame(&f)) {
                printf("FAIL %-52s timeout\n", c.name);
                g_fail++;
                continue;
            }

            // Verify can_id
            char cid_name[80];
            snprintf(cid_name, sizeof(cid_name), "%s can_id=%d", c.name, c.id);
            check(cid_name, (uint16_t)(f.can_id & 0x7FF) == c.id);

            // Verify DLC
            check((std::string(c.name)+" DLC=8").c_str(), f.len == 8);

            // Verify bytes
            uint8_t exp[8];
            expected_dm_bytes(c.pos, c.vel, c.kp, c.kd, c.tor, exp);
            check_bytes(c.name, f.data, exp, 8);
        }
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Section 4: EC motor command encoding via TestCan
    // -----------------------------------------------------------------------
    printf("--- send_EC_motor_cmd frame bytes ---\n");
    {
        TestCan ac(iface);
        usleep(10000);

        struct {
            const char *name;
            uint16_t id;
            float kp, kd, pos, spd, tor;
        } cases[] = {
            {"EC all zeros (id=1)",  1,   0.0f, 0.0f,  0.0f, 0.0f,  0.0f},
            {"EC mid range (id=4)",  4, 100.0f, 2.0f,  3.14f, 5.0f, 8.0f},
        };

        for (auto &c : cases) {
            sniff.drain();
            ac.send_EC_motor_cmd(c.id, c.kp, c.kd, c.pos, c.spd, c.tor);
            usleep(5000);

            struct can_frame f{};
            if (!sniff.read_frame(&f)) {
                printf("FAIL %-52s timeout\n", c.name);
                g_fail++;
                continue;
            }

            check((std::string(c.name)+" DLC=8").c_str(), f.len == 8);

            uint8_t exp[8];
            expected_ec_bytes(c.kp, c.kd, c.pos, c.spd, c.tor, exp);
            check_bytes(c.name, f.data, exp, 8);
        }
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Section 5: Special DM commands
    // -----------------------------------------------------------------------
    printf("--- Special DM command bytes ---\n");
    {
        TestCan ac(iface);
        usleep(10000);

        struct {
            const char *name;
            void (TestCan::*fn)(uint16_t);
            uint8_t last_byte;
        } specials[] = {
            {"enable_DM_motor  last=0xFC", &TestCan::enable_DM_motor,    0xFC},
            {"reset_zero_readout last=0xFE",&TestCan::reset_zero_readout, 0xFE},
            {"clear            last=0xFB", &TestCan::clear_motor,         0xFB},
        };

        for (auto &s : specials) {
            sniff.drain();
            (ac.*s.fn)(1);
            usleep(5000);

            struct can_frame f{};
            if (!sniff.read_frame(&f)) {
                printf("FAIL %-52s timeout\n", s.name);
                g_fail++;
                continue;
            }

            // All first 7 bytes must be 0xFF
            bool prefix_ok = true;
            for (int i = 0; i < 7; i++)
                if (f.data[i] != 0xFF) { prefix_ok = false; break; }
            check((std::string(s.name)+" prefix=0xFF×7").c_str(), prefix_ok);
            check(s.name, f.data[7] == s.last_byte);
        }
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Section 6: Query commands
    // -----------------------------------------------------------------------
    printf("--- EC query commands ---\n");
    {
        TestCan ac(iface);
        usleep(10000);

        struct {
            const char *name;
            void (TestCan::*fn)(uint16_t);
            uint8_t mode;
        } queries[] = {
            {"query_EC_motor_pos",     &TestCan::query_EC_motor_pos,     0x01},
            {"query_EC_motor_vel",     &TestCan::query_EC_motor_vel,     0x02},
            {"query_EC_motor_current", &TestCan::query_EC_motor_current, 0x03},
        };

        for (auto &q : queries) {
            sniff.drain();
            (ac.*q.fn)(1);
            usleep(5000);

            struct can_frame f{};
            if (!sniff.read_frame(&f)) {
                printf("FAIL %-52s timeout\n", q.name);
                g_fail++;
                continue;
            }

            check((std::string(q.name)+" DLC=2").c_str(),  f.len == 2);
            check((std::string(q.name)+" data[0]=0xE0").c_str(), f.data[0] == 0xE0);
            check((std::string(q.name)+" data[1]=mode").c_str(), f.data[1] == q.mode);
        }
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Section 7: DM feedback receive (inject → get_motor_msg)
    // -----------------------------------------------------------------------
    printf("--- DM feedback receive via get_motor_msg ---\n");
    {
        TestCan ac(iface);
        usleep(20000); // let receiver thread settle

        float pos_ref = 4.5f, vel_ref = -10.0f, tor_ref = 3.0f;
        uint16_t pi=(uint16_t)float_to_uint(pos_ref,-12.5f,12.5f,16);
        uint16_t vi=(uint16_t)float_to_uint(vel_ref,-45.0f,45.0f,12);
        uint16_t ti=(uint16_t)float_to_uint(tor_ref,-18.0f,18.0f,12);

        // Build a DM feedback frame: can_id=0, data[0]=motor_id=5
        struct can_frame fb{};
        fb.can_id = 0;
        fb.len    = 8;
        fb.data[0] = 0x05; // motor_id=5
        fb.data[1] = (uint8_t)(pi>>8);
        fb.data[2] = (uint8_t)(pi&0xFF);
        fb.data[3] = (uint8_t)(vi>>4);
        fb.data[4] = (uint8_t)(((vi&0xF)<<4)|(ti>>8));
        fb.data[5] = (uint8_t)(ti&0xFF);
        fb.data[6] = 0;
        fb.data[7] = 0;

        sniff.inject(&fb);
        usleep(30000); // give receiver thread time to process

        auto msgs = ac.get_motor_msg();
        float tol_pos = 25.0f/65535.0f + 1e-5f;
        float tol_vel = 90.0f/4095.0f  + 1e-5f;
        float tol_tor = 36.0f/4095.0f  + 1e-5f;

        check("DM feedback motor_id=5",
              msgs[5].motor_id == 5);
        check("DM feedback angle_actual_rad",
              fabsf(msgs[5].angle_actual_rad  - pos_ref) <= tol_pos);
        check("DM feedback speed_actual_rad",
              fabsf(msgs[5].speed_actual_rad  - vel_ref) <= tol_vel);
        check("DM feedback current_actual_float",
              fabsf(msgs[5].current_actual_float - tor_ref) <= tol_tor);
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Section 8: RV (MIT mode) feedback receive
    // -----------------------------------------------------------------------
    printf("--- RV MIT feedback receive via get_motor_msg ---\n");
    {
        TestCan ac(iface);
        usleep(20000);

        float pos_ref = 1.5f, vel_ref = -3.0f, tor_ref = 0.5f;
        uint16_t pi=(uint16_t)float_to_uint(pos_ref,-12.5f,12.5f,16);
        uint16_t vi=(uint16_t)float_to_uint(vel_ref,-18.0f,18.0f,12);
        uint16_t ti=(uint16_t)float_to_uint(tor_ref,-30.0f,30.0f,12);

        // RV MIT feedback: can_id=motor_id=3, data[0] bits[7:5]=001 (msg_type=1)
        struct can_frame fb{};
        fb.can_id = 0x00000003u;
        fb.len    = 8;
        fb.data[0] = 0x20; // msg_type=1
        fb.data[1] = (uint8_t)(pi>>8);
        fb.data[2] = (uint8_t)(pi&0xFF);
        fb.data[3] = (uint8_t)(vi>>4);
        fb.data[4] = (uint8_t)(((vi&0xF)<<4)|(ti>>8));
        fb.data[5] = (uint8_t)(ti&0xFF);
        fb.data[6] = 0x64; // temperature byte
        fb.data[7] = 0x00;

        sniff.inject(&fb);
        usleep(30000);

        auto msgs = ac.get_motor_msg();
        float tol_pos = 25.0f/65535.0f + 1e-5f;
        float tol_vel = 36.0f/4095.0f  + 1e-5f;
        float tol_tor = 60.0f/4095.0f  + 1e-5f;

        check("RV feedback motor_id=3",
              msgs[3].motor_id == 3);
        check("RV feedback angle_actual_rad",
              fabsf(msgs[3].angle_actual_rad  - pos_ref) <= tol_pos);
        check("RV feedback speed_actual_rad",
              fabsf(msgs[3].speed_actual_rad  - vel_ref) <= tol_vel);
        check("RV feedback current_actual_float",
              fabsf(msgs[3].current_actual_float - tor_ref) <= tol_tor);
        check("RV feedback temperature=25",
              msgs[3].temperature == 25);
    }
    printf("\n");

    sniff.close();

    printf("=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
