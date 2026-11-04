/* aux_tenant.hip -- a simulated tenant of the aux worker's extra card for window arms with an AxF part
 * (work/aux-dyn/SPEC.md §8.4). Run only in a GPU window, only with the extra card visible
 * (ROCR_VISIBLE_DEVICES=<its UUID>, HIP_VISIBLE_DEVICES=0); never by a host test.
 *
 *   aux_tenant MIB HOLD_S WAIT_S
 *
 * hipInit, 64 MiB allocated and touched (a client of the card from here on: over any _SHARE_MIN_MIB up to 64), then
 * "aux_tenant: open <unix ms>". A polite tenant: it polls hipMemGetInfo every 50 ms until MIB + 256 MiB are free (the
 * server's release) or WAIT_S pass (exit 2, "no room"); then MIB allocated and set, "aux_tenant: room after <ms> ms,
 * holding <MIB> MiB" (from the open line); it holds them HOLD_S s (0: until SIGTERM, which it takes as the end), frees
 * them, "aux_tenant: freed", exit 0. Its comm is aux_tenant (the harness exempts it from an arm's client checks). */
#include <hip/hip_runtime.h>

#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {
volatile sig_atomic_t g_term = 0;
void on_term(int) { g_term = 1; }
long long unix_ms() {
    return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s MIB HOLD_S WAIT_S\n", argv[0]);
        return 64;
    }
    const long long mib = std::atoll(argv[1]), hold_s = std::atoll(argv[2]), wait_s = std::atoll(argv[3]);
    if (mib < 1 || hold_s < 0 || wait_s < 0) {
        std::fprintf(stderr, "aux_tenant: MIB >= 1, HOLD_S >= 0, WAIT_S >= 0\n");
        return 64;
    }
    struct sigaction sa {};
    sa.sa_handler = on_term;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
    if (hipInit(0) != hipSuccess || hipSetDevice(0) != hipSuccess) {
        std::fprintf(stderr, "aux_tenant: no device\n");
        return 1;
    }
    void* small = nullptr;
    const size_t small_b = (size_t)64 << 20;
    if (hipMalloc(&small, small_b) != hipSuccess || hipMemset(small, 0, small_b) != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) {
        std::fprintf(stderr, "aux_tenant: the first 64 MiB were refused\n");
        return 1;
    }
    const long long t_open = unix_ms();
    std::printf("aux_tenant: open %lld\n", t_open);
    std::fflush(stdout);
    const size_t want = (size_t)mib << 20;
    for (;;) {
        size_t free_b = 0, total_b = 0;
        if (hipMemGetInfo(&free_b, &total_b) == hipSuccess && free_b >= want + ((size_t)256 << 20)) break;
        if (g_term || unix_ms() - t_open >= wait_s * 1000) {
            std::printf("aux_tenant: no room for %lld MiB after %lld ms (%zu MiB free)\n", mib, unix_ms() - t_open,
                        free_b >> 20);
            (void)hipFree(small);
            return 2;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    void* big = nullptr;
    if (hipMalloc(&big, want) != hipSuccess || hipMemset(big, 1, want) != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) {
        std::printf("aux_tenant: %lld MiB refused after the room was seen\n", mib);
        (void)hipFree(small);
        return 3;
    }
    std::printf("aux_tenant: room after %lld ms, holding %lld MiB\n", unix_ms() - t_open, mib);
    std::fflush(stdout);
    const long long t_hold = unix_ms();
    while (!g_term && (hold_s == 0 || unix_ms() - t_hold < hold_s * 1000))
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    (void)hipFree(big);
    (void)hipFree(small);
    std::printf("aux_tenant: freed\n");
    std::fflush(stdout);
    return 0;
}
