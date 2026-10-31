/* Fresh-process check: the core and plugin cache their LCP environment readers once. No device,
 * model, backend or emulator is linked; each CTest entry tests one real startup value. */
#include "../core/place/ep_route_counts.h"
#include "../libr4d/r4d_ds41_tall.h"
#include <cstdio>
#include <cstdlib>

static_assert(rad::kEpRouteLcpMin == kR4dDs41LcpMin && rad::kEpRouteLcpMax == kR4dDs41LcpMax,
              "core and kernel LCP ranges must agree");
static_assert(kR4dDs41LcpMax <= kR4dDs41P1MaxRows, "P1 must serve the admitted chunk");

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const int core = std::atoi(argv[1]), plugin = std::atoi(argv[2]);
    const int64_t got_core = rad::ds41_lcp_state();
    const int got_plugin = r4d_ds41_lcp();
    const bool pins = r4d_ds41_exact_tall();
    const bool ok = got_core == core && got_plugin == plugin && pins == (plugin > 0) &&
                    (plugin <= 0 || r4d_ds41_tall_rows() >= plugin);
    std::printf("%s: LCP startup core=%lld plugin=%d pins=%d; expected %d/%d\n", ok ? "PASS" : "FAIL",
                (long long)got_core, got_plugin, pins, core, plugin);
    return ok ? 0 : 1;
}
