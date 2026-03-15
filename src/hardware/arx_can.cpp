#include "hardware/arx_can.h"
#include "libcan/SocketCAN.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <array>
#include <pthread.h>
#include <unistd.h>

// SOEM headers (EtherCAT)
extern "C" {
#include <soem/ethercat.h>
}

// ---------------------------------------------------------------------------
// Helpers: CAN protocol data unpacking (RV / DM motor feedback frames)
// ---------------------------------------------------------------------------

void RV_can_data_repack(uint32_t msgID, uint8_t *data, int32_t databufferlen,
                        uint8_t comm_mode, std::array<OD_Motor_Msg, 10> &motor_msg)
{
    (void)databufferlen;
    if (comm_mode != 0)
        return;

    uint8_t id = msgID & 0xFF;
    if (id >= 10) return;

    uint8_t msg_type = data[0] >> 5;

    motor_msg[id].motor_id = id;

    if (msg_type == 1) {
        // MIT mode reply: pos (16-bit), vel (12-bit), tor (12-bit), temp, error
        uint16_t pos_int = ((uint16_t)data[1] << 8) | data[2];
        uint16_t vel_int = ((uint16_t)data[3] << 4) | (data[4] >> 4);
        uint16_t tor_int = ((uint16_t)(data[4] & 0x0F) << 8) | data[5];

        motor_msg[id].angle_actual_rad  = uint_to_float(pos_int, -12.5f, 12.5f, 16);
        motor_msg[id].speed_actual_rad  = uint_to_float(vel_int, -18.0f, 18.0f, 12);
        motor_msg[id].current_actual_float = uint_to_float(tor_int, -30.0f, 30.0f, 12);
        motor_msg[id].temperature = (data[6] - 50) / 2;
        motor_msg[id].error = data[0] & 0x1F;

    } else if (msg_type == 2) {
        // Position / velocity / current query reply
        // Bytes 1..4: float (little-endian) for angle
        uint32_t raw;
        raw  = (uint32_t)data[3] | ((uint32_t)data[2] << 8) |
               ((uint32_t)data[1] << 16);          // 3-byte big-endian float mantissa
        motor_msg[id].angle_actual_float = *reinterpret_cast<float *>(&raw);

        uint16_t current_int = ((uint16_t)data[5] << 8) | data[6];
        motor_msg[id].current_actual_int = (int16_t)current_int;
        motor_msg[id].speed_actual_rad   = (float)(int16_t)current_int / 100.0f;
        motor_msg[id].temperature = (data[7] - 50) / 2;
        motor_msg[id].error = data[0] & 0x1F;
    }
}

void DM_can_data_repack(uint8_t *data, std::array<OD_Motor_Msg, 10> &motor_msg)
{
    uint8_t id = data[0] & 0x0F;
    if (id >= 10) return;

    uint16_t pos_int = ((uint16_t)data[1] << 8) | data[2];
    uint16_t vel_int = ((uint16_t)data[3] << 4) | (data[4] >> 4);
    uint16_t tor_int = ((uint16_t)(data[4] & 0x0F) << 8) | data[5];

    motor_msg[id].motor_id = id;
    motor_msg[id].angle_actual_rad     = uint_to_float(pos_int, -12.5f, 12.5f, 16);
    motor_msg[id].speed_actual_rad     = uint_to_float(vel_int, -45.0f, 45.0f, 12);
    motor_msg[id].current_actual_float = uint_to_float(tor_int, -18.0f, 18.0f, 12);
}

// ---------------------------------------------------------------------------
// Callback used by SocketCAN receiver thread for Usb2Can
// ---------------------------------------------------------------------------
void can_ReceiveHandlerProxy(can_frame_t *frame, void *arg)
{
    CanInterface *iface = reinterpret_cast<CanInterface *>(arg);
    iface->can_receive_frame(frame);
}

// ---------------------------------------------------------------------------
// CanInterface
// ---------------------------------------------------------------------------
void CanInterface::can_receive_frame(can_frame_t *frame)
{
    if (frame->can_id == 0) {
        // DM motor frame (id = 0 means motor_id embedded in data[0])
        std::lock_guard<std::mutex> lock(motor_msg_mutex_);
        DM_can_data_repack(frame->data, motor_msg_);
    } else {
        uint32_t motor_id = frame->can_id & 0xFF;
        if (motor_id == 0 || motor_id > 7) {
            throw std::runtime_error("invalid can_id");
        }
        std::lock_guard<std::mutex> lock(motor_msg_mutex_);
        RV_can_data_repack(frame->can_id, frame->data, frame->len, 0, motor_msg_);
    }
}

const std::array<OD_Motor_Msg, 10> CanInterface::get_motor_msg()
{
    std::lock_guard<std::mutex> lock(motor_msg_mutex_);
    return motor_msg_;
}

// ---------------------------------------------------------------------------
// Usb2Can
// ---------------------------------------------------------------------------
Usb2Can::Usb2Can(std::string interface_name)
    : interface_name_(std::move(interface_name))
{
    socketcan_.reception_handler_data = this;
    socketcan_.reception_handler      = can_ReceiveHandlerProxy;
    socketcan_.open(interface_name_.c_str());
}

Usb2Can::~Usb2Can()
{
    socketcan_.close();
}

void Usb2Can::transmit(can_frame_t &frame)
{
    if (!socketcan_.is_open()) {
        puts("Unable to transmit: Socket not open");
        return;
    }
    socketcan_.transmit(&frame);
}

bool Usb2Can::is_open()
{
    return socketcan_.is_open();
}

// ---------------------------------------------------------------------------
// EtherCat2Can — wraps SOEM EtherCAT master
// IOmap layout (from decompiled sizes):
//   Output (master→slave): 30 bytes  — EcatTxPacket
//   Input  (slave→master): 34 bytes  — EcatRxPacket + 2 bytes overhead
// ---------------------------------------------------------------------------

// Full definition of the forward-declared EcatState (matches header forward decl)
struct EcatState {
    ecx_contextt  ctx;
    char          iface[IFNAMSIZ];
    uint8_t       iomap[4096];
    int           slave_idx = 1;
    int           wkc_expected = 0;
    int           counter = 0;
};

EtherCat2Can::EtherCat2Can(std::string interface_name)
    : interface_name_(std::move(interface_name)),
      ecat_(new EcatState())
{
    EcatState &ec = *ecat_;
    std::strncpy(ec.iface, interface_name_.c_str(), IFNAMSIZ - 1);

    // Initialize SOEM context (uses the global ec_* variables internally)
    if (ecx_init(&ec.ctx, ec.iface) <= 0) {
        throw std::runtime_error(
            std::string("Failed to init EtherCAT on interface ") + interface_name_);
    }

    if (ecx_config_init(&ec.ctx, 0) <= 0) {
        ecx_close(&ec.ctx);
        throw std::runtime_error("No EtherCAT slaves found on " + interface_name_);
    }

    ecx_config_map_group(&ec.ctx, ec.iomap, 0);
    ecx_configdc(&ec.ctx);

    // Request OP state
    ec.ctx.slavelist[0].state = EC_STATE_OPERATIONAL;
    ecx_writestate(&ec.ctx, 0);

    // Wait up to 5 s for slaves to reach OP
    int chk = 200;
    do {
        ecx_send_processdata(&ec.ctx);
        ecx_receive_processdata(&ec.ctx, EC_TIMEOUTRET);
        ecx_statecheck(&ec.ctx, 0, EC_STATE_OPERATIONAL, 50000);
    } while (chk-- > 0 && ec.ctx.slavelist[0].state != EC_STATE_OPERATIONAL);

    if (ec.ctx.slavelist[0].state != EC_STATE_OPERATIONAL) {
        ecx_close(&ec.ctx);
        throw std::runtime_error(
            std::string("Failed to bring EtherCAT slave to OP state on ") + interface_name_);
    }

    // Expected working counter: one slave with in + out
    ec.wkc_expected = (ec.ctx.grouplist[0].outputsWKC * 2) +
                       ec.ctx.grouplist[0].inputsWKC;
    printf("EtherCAT ready on %s, wkc_expected=%d\n",
           interface_name_.c_str(), ec.wkc_expected);
}

EtherCat2Can::~EtherCat2Can()
{
    if (!ecat_) return;
    EcatState &ec = *ecat_;
    printf("Closing EtherCAT %s\n", interface_name_.c_str());
    ec.ctx.slavelist[0].state = EC_STATE_INIT;
    ecx_writestate(&ec.ctx, 0);
    printf("EtherCAT state: %d\n", (int)ec.ctx.slavelist[0].state);
    ecx_close(&ec.ctx);
    puts("EtherCAT closed");
}

void EtherCat2Can::transmit(can_frame_t &frame)
{
    EcatState &ec = *ecat_;

    // Check slave is still in OP
    ecx_readstate(&ec.ctx);
    if (ec.ctx.slavelist[ec.slave_idx].state != EC_STATE_OPERATIONAL) {
        puts("EtherCAT slave not operational");
        uint16_t al = ec.ctx.slavelist[ec.slave_idx].ALstatuscode;
        printf("Slave %d State=0x%2.2x StatusCode=0x%4.4x: %s\n",
               ec.slave_idx,
               (unsigned)ec.ctx.slavelist[ec.slave_idx].state,
               (unsigned)al,
               ec_ALstatuscode2string(al));
        return;
    }

    // Build output PDO: EcatTxPacket
    EcatTxPacket tx;
    std::memset(&tx, 0, sizeof(tx));
    tx.LED = 1;
    tx.can[0].StdId = (uint16_t)(frame.can_id & 0x7FF);
    tx.can[0].DLC   = frame.len;
    std::memcpy(tx.can[0].Data, frame.data, frame.len);

    uint8_t *outputs = ec.ctx.slavelist[ec.slave_idx].outputs;
    std::memcpy(outputs, &tx, sizeof(tx));

    ecx_send_processdata(&ec.ctx);
    int wkc = ecx_receive_processdata(&ec.ctx, EC_TIMEOUTRET);

    EcatRxPacket rx;
    std::memset(&rx, 0, sizeof(rx));
    if (wkc >= ec.wkc_expected) {
        uint8_t *inputs = ec.ctx.slavelist[ec.slave_idx].inputs;
        std::memcpy(&rx, inputs, sizeof(rx));
    } else {
        printf("wkc %d\n", wkc);
    }

    // Decode received CAN frames from PDO
    can_frame_t rx_frame;
    std::memset(&rx_frame, 0, sizeof(rx_frame));
    rx_frame.can_id = rx.can[0].StdId;
    rx_frame.len    = rx.can[0].DLC;
    std::memcpy(rx_frame.data, rx.can[0].Data, rx.can[0].DLC);
    can_receive_frame(&rx_frame);

    ec.counter++;
}

bool EtherCat2Can::is_open()
{
    if (!ecat_) return false;
    EcatState &ec = *ecat_;
    ecx_readstate(&ec.ctx);
    return ec.ctx.slavelist[0].state == EC_STATE_OPERATIONAL;
}

// ---------------------------------------------------------------------------
// ArxCan — thin wrapper: selects Usb2Can or EtherCat2Can by interface name
// ---------------------------------------------------------------------------
ArxCan::ArxCan(std::string interface_name)
{
    // "can…" → USB-CAN adapter (SocketCAN)
    // "en…" or "eth…" → EtherCAT-CAN adapter
    if (interface_name.find("can") == 0) {
        can_interface_ = std::make_shared<Usb2Can>(interface_name);
    } else if (interface_name.find("en") == 0 ||
               interface_name.find("eth") == 0) {
        can_interface_ = std::make_shared<EtherCat2Can>(interface_name);
    } else {
        throw std::runtime_error(
            "Interface name should start with can (for USB-CAN adapter) or "
            "en/eth (for EtherCAT-CAN adapter). Unrecognized interface name " +
            interface_name);
    }

    if (!can_interface_->is_open()) {
        throw std::runtime_error(
            "Failed to open CAN adapter with interface name " + interface_name);
    }
}

ArxCan::~ArxCan()
{
    can_interface_.reset();
}

// ---------------------------------------------------------------------------
// ArxCan motor commands
// ---------------------------------------------------------------------------

// --- EC (RV) motor ---
// CAN frame data layout (8 bytes):
//   data[0]    = kp[11:7]          (5 bits in [4:0])
//   data[1]    = kp[6:0]<<1 | kd[8]
//   data[2]    = kd[7:0]
//   data[3]    = pos[15:8]
//   data[4]    = pos[7:0]
//   data[5]    = vel[11:4]
//   data[6]    = vel[3:0]<<4 | tor[11:8]
//   data[7]    = tor[7:0]

void ArxCan::send_EC_motor_cmd(uint16_t motor_id, float kp, float kd,
                                float pos, float spd, float tor)
{
    // Clamp
    kp  = fmaxf(0.0f,   fminf(500.0f, kp));
    kd  = fmaxf(0.0f,   fminf(5.0f,   kd));
    pos = fmaxf(-12.5f, fminf(12.5f,  pos));
    spd = fmaxf(-18.0f, fminf(18.0f,  spd));
    tor = fmaxf(-30.0f, fminf(30.0f,  tor));

    uint16_t kp_int  = (uint16_t)float_to_uint(kp,   0.0f,   500.0f, 12);
    uint16_t kd_int  = (uint16_t)float_to_uint(kd,   0.0f,     5.0f,  9);
    uint16_t pos_int = (uint16_t)float_to_uint(pos, -12.5f,   12.5f, 16);
    uint16_t vel_int = (uint16_t)float_to_uint(spd, -18.0f,   18.0f, 12);
    uint16_t tor_int = (uint16_t)float_to_uint(tor, -30.0f,   30.0f, 12);

    can_frame_t frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.can_id = motor_id;
    frame.len    = 8;
    frame.data[0] = (uint8_t)(kp_int >> 7);
    frame.data[1] = (uint8_t)(((kp_int & 0x7F) << 1) | (kd_int >> 8));
    frame.data[2] = (uint8_t)(kd_int & 0xFF);
    frame.data[3] = (uint8_t)(pos_int >> 8);
    frame.data[4] = (uint8_t)(pos_int & 0xFF);
    frame.data[5] = (uint8_t)(vel_int >> 4);
    frame.data[6] = (uint8_t)(((vel_int & 0x0F) << 4) | (tor_int >> 8));
    frame.data[7] = (uint8_t)(tor_int & 0xFF);

    can_interface_->transmit(frame);
}

// --- DM motor ---
// CAN frame data layout (8 bytes):
//   data[0] = pos[15:8]
//   data[1] = pos[7:0]
//   data[2] = vel[11:4]
//   data[3] = vel[3:0]<<4 | kp[11:8]
//   data[4] = kp[7:0]
//   data[5] = kd[11:4]
//   data[6] = kd[3:0]<<4 | tor[11:8]
//   data[7] = tor[7:0]

void ArxCan::send_DM_motor_cmd(uint16_t motor_id, float kp, float kd,
                                float pos, float spd, float tor)
{
    // Clamp
    kp  = fmaxf(0.0f,   fminf(500.0f, kp));
    kd  = fmaxf(0.0f,   fminf(5.0f,   kd));
    pos = fmaxf(-12.5f, fminf(12.5f,  pos));
    spd = fmaxf(-45.0f, fminf(45.0f,  spd));
    tor = fmaxf(-18.0f, fminf(18.0f,  tor));

    uint16_t pos_int = (uint16_t)float_to_uint(pos, -12.5f, 12.5f,  16);
    uint16_t vel_int = (uint16_t)float_to_uint(spd, -45.0f, 45.0f,  12);
    uint16_t kp_int  = (uint16_t)float_to_uint(kp,   0.0f, 500.0f,  12);
    uint16_t kd_int  = (uint16_t)float_to_uint(kd,   0.0f,   5.0f,  12);
    uint16_t tor_int = (uint16_t)float_to_uint(tor, -18.0f, 18.0f,  12);

    can_frame_t frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.can_id = motor_id;
    frame.len    = 8;
    frame.data[0] = (uint8_t)(pos_int >> 8);
    frame.data[1] = (uint8_t)(pos_int & 0xFF);
    frame.data[2] = (uint8_t)(vel_int >> 4);
    frame.data[3] = (uint8_t)(((vel_int & 0x0F) << 4) | (kp_int >> 8));
    frame.data[4] = (uint8_t)(kp_int & 0xFF);
    frame.data[5] = (uint8_t)(kd_int >> 4);
    frame.data[6] = (uint8_t)(((kd_int & 0x0F) << 4) | (tor_int >> 8));
    frame.data[7] = (uint8_t)(tor_int & 0xFF);

    can_interface_->transmit(frame);
}

void ArxCan::enable_DM_motor(uint16_t motor_id)
{
    can_frame_t frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.can_id = motor_id;
    frame.len    = 8;
    // Enable command: 0xFF FF FF FF FF FF FF FC
    std::memset(frame.data, 0xFF, 8);
    frame.data[7] = 0xFC;
    can_interface_->transmit(frame);
}

void ArxCan::reset_zero_readout(uint16_t motor_id)
{
    can_frame_t frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.can_id = motor_id;
    frame.len    = 8;
    std::memset(frame.data, 0xFF, 8);
    frame.data[7] = 0xFE;
    can_interface_->transmit(frame);
}

void ArxCan::clear(uint16_t motor_id)
{
    can_frame_t frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.can_id = motor_id;
    frame.len    = 8;
    std::memset(frame.data, 0xFF, 8);
    frame.data[7] = 0xFB;
    can_interface_->transmit(frame);
}

void ArxCan::set_motor(uint16_t motor_id, uint8_t cmd)
{
    if (cmd == 0) return;

    can_frame_t frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.can_id  = 0x7FF;
    frame.len     = 4;
    frame.data[0] = (uint8_t)(motor_id >> 8);
    frame.data[1] = (uint8_t)(motor_id & 0xFF);
    frame.data[2] = 0x00;
    frame.data[3] = cmd;
    can_interface_->transmit(frame);
}

void ArxCan::can_cmd_init(uint16_t motor_id, uint8_t cmd)
{
    set_motor(motor_id, cmd);

    // Transmit an empty MIT-mode frame to complete initialization
    can_frame_t frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.can_id = motor_id;
    frame.len    = 8;
    can_interface_->transmit(frame);
}

// EC motor query commands:
//   data[0] = 0xE0, data[1] = mode (0x01=pos, 0x02=vel, 0x03=current)

void ArxCan::query_EC_motor_pos(uint16_t motor_id)
{
    can_frame_t frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.can_id  = motor_id;
    frame.len     = 2;
    frame.data[0] = 0xE0;
    frame.data[1] = 0x01;
    can_interface_->transmit(frame);
}

void ArxCan::query_EC_motor_vel(uint16_t motor_id)
{
    can_frame_t frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.can_id  = motor_id;
    frame.len     = 2;
    frame.data[0] = 0xE0;
    frame.data[1] = 0x02;
    can_interface_->transmit(frame);
}

void ArxCan::query_EC_motor_current(uint16_t motor_id)
{
    can_frame_t frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.can_id  = motor_id;
    frame.len     = 2;
    frame.data[0] = 0xE0;
    frame.data[1] = 0x03;
    can_interface_->transmit(frame);
}

const std::array<OD_Motor_Msg, 10> ArxCan::get_motor_msg()
{
    return can_interface_->get_motor_msg();
}
