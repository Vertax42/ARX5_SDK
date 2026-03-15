#include "libcan/SocketCAN.h"
#include "libcan/CANFrame.h"

#include <cstdio>
#include <cstring>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#include <sys/select.h>

// CAN_RAW is defined in CANFrame.h (value 1)
#ifndef CAN_RAW
#define CAN_RAW 1
#endif

// ---------------------------------------------------------------------------
// Receiver thread: reads CAN frames and dispatches them via reception_handler
// ---------------------------------------------------------------------------
static void *socketcan_receiver_thread(void *arg)
{
    SocketCAN *can = reinterpret_cast<SocketCAN *>(arg);
    can->receiver_thread_running = true;

    can_frame_t frame;
    while (!can->terminate_receiver_thread) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(can->sockfd, &readfds);

        struct timeval timeout;
        timeout.tv_sec  = 0;
        timeout.tv_usec = 100000; // 100 ms

        int ret = select(can->sockfd + 1, &readfds, nullptr, nullptr, &timeout);
        if (ret > 0 && FD_ISSET(can->sockfd, &readfds)) {
            ssize_t nbytes = read(can->sockfd, &frame, sizeof(frame));
            if (nbytes > 0 && can->reception_handler != nullptr) {
                can->reception_handler(&frame, can->reception_handler_data);
            }
        }
    }

    can->receiver_thread_running = false;
    return nullptr;
}

// ---------------------------------------------------------------------------
// SocketCAN
// ---------------------------------------------------------------------------
SocketCAN::SocketCAN()
    : adapter_type(ADAPTER_SOCKETCAN),
      reception_handler_data(nullptr),
      reception_handler(nullptr),
      sockfd(-1),
      terminate_receiver_thread(false),
      receiver_thread_running(false)
{
    std::memset(&if_request, 0, sizeof(if_request));
    std::memset(&addr, 0, sizeof(addr));
    std::memset(&receiver_thread_id, 0, sizeof(receiver_thread_id));
    puts("SocketCAN adapter created.");
}

SocketCAN::~SocketCAN()
{
    puts("Destroying SocketCAN adapter...");
    if (is_open()) {
        close();
    }
}

void SocketCAN::open(const char *interface_name)
{
    int fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    sockfd = fd;
    if (fd == -1) {
        puts("Error: Unable to create a CAN socket");
        return;
    }

    printf("Created CAN socket with descriptor %d.\n", fd);

    std::strncpy(if_request.ifr_name, interface_name, IFNAMSIZ - 1);
    if (ioctl(sockfd, SIOCGIFINDEX, &if_request) == -1) {
        printf("Unable to select CAN interface %s: I/O control error\n", interface_name);
        goto cleanup;
    }

    printf("Found: %s has interface index %d.\n", interface_name, if_request.ifr_ifindex);

    addr.can_family  = AF_CAN;
    addr.can_ifindex = if_request.ifr_ifindex;

    if (bind(sockfd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) == -1) {
        puts("Failed to bind socket to network interface");
        goto cleanup;
    }

    printf("Successfully bound socket to interface %d.\n", if_request.ifr_ifindex);

    terminate_receiver_thread = false;
    if (pthread_create(&receiver_thread_id, nullptr, socketcan_receiver_thread, this) == 0) {
        printf("Successfully started receiver thread with ID %d.\n",
               (int)receiver_thread_id);
        return;
    }
    puts("Unable to start receiver thread.");
    return;

cleanup:
    puts("Waiting for receiver thread to terminate.");
    terminate_receiver_thread = true;
    while (receiver_thread_running) {
        sleep(1);
    }
    if (is_open()) {
        ::close(sockfd);
        sockfd = -1;
    }
    puts("CAN socket destroyed.");
}

void SocketCAN::close()
{
    puts("Waiting for receiver thread to terminate.");
    terminate_receiver_thread = true;
    while (receiver_thread_running) {
        sleep(1);
    }
    if (is_open()) {
        ::close(sockfd);
        sockfd = -1;
        puts("CAN socket destroyed.");
    }
}

bool SocketCAN::is_open()
{
    return sockfd != -1;
}

void SocketCAN::transmit(can_frame_t *frame)
{
    if (!is_open()) {
        puts("Unable to transmit: Socket not open");
        return;
    }
    write(sockfd, frame, sizeof(can_frame_t));
}

void SocketCAN::start_receiver_thread()
{
    terminate_receiver_thread = false;
    if (pthread_create(&receiver_thread_id, nullptr, socketcan_receiver_thread, this) == 0) {
        printf("Successfully started receiver thread with ID %d.\n",
               (int)receiver_thread_id);
        return;
    }
    puts("Unable to start receiver thread.");
}
