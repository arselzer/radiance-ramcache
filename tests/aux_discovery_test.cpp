/* aux_discovery_test.cpp -- the aux expert worker's switches, selector and discovery decision
 * (tools/ds41-host-experts/ds41_aux.h; work/aux/SPEC.md §2-§3), pure:
 *  - the switch: off when unset, "" or "0" (nothing else read), on at 1024..1048576 MiB, refused otherwise;
 *    the policy, headroom, deadline, backoff and forced-lateness ranges; the selector's BDF and UUID forms;
 *  - PCI path parsing: the root complex and port, unknown;
 *  - decide(): no candidate, wrong ISA, integrated, too little free VRAM (and exactly enough), a mixed TP set
 *    with a same-arch leftover (the WARN), the selector naming a rank's card, nothing, a BDF or a UUID (a fake
 *    KFD topology), and the choice between two candidates (independent root, then free VRAM, then BDF). */
#include "rad_test.h"
#include "place/aux_expert_worker.h"

#include <sys/stat.h>

#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace ds41aux;

namespace {

struct Env {
    std::map<std::string, std::string> v;
    mutable std::vector<std::string> read;
    const char* get(const char* k) const {
        read.push_back(k);
        auto it = v.find(k);
        return it == v.end() ? nullptr : it->second.c_str();
    }
};
int parse_env(const Env& e, Config* c, std::string* why) {
    return parse([&](const char* k) { return e.get(k); }, c, why);
}

constexpr int64_t GiB = (int64_t)1 << 30;
Card card(int ord, const char* name, const char* arch, const char* bdf, const char* complex, const char* port,
          int64_t free_bytes) {
    Card c;
    c.ord = ord;
    c.name = name;
    c.arch = arch;
    c.bdf = bdf;
    c.complex = complex;
    c.port = port;
    c.vram_bytes = 32 * GiB;
    c.vram_free = free_bytes;
    return c;
}
const char* kXT = "AMD Radeon RX 9070 XT";
const char* kR9700 = "AMD Radeon AI PRO R9700";
/* The box: four XTs (ranks), the R9700 on root complex c0 with rank 3's card on another port. */
std::vector<Card> box(int64_t r9700_free) {
    return { card(0, kXT, "gfx1201", "0000:03:00.0", "pci0000:00", "0000:00:01.1", 1 * GiB),
             card(1, kXT, "gfx1201", "0000:48:00.0", "pci0000:40", "0000:40:01.1", 1 * GiB),
             card(2, kXT, "gfx1201", "0000:83:00.0", "pci0000:80", "0000:80:01.1", 1 * GiB),
             card(3, kXT, "gfx1201", "0000:c6:00.0", "pci0000:c0", "0000:c0:03.1", 1 * GiB),
             card(4, kR9700, "gfx1201", "0000:c3:00.0", "pci0000:c0", "0000:c0:01.1", r9700_free) };
}
const std::vector<int> kRanks = { 0, 1, 2, 3 };
Config on(int64_t cap = 26000) {
    Config c;
    c.on = true;
    c.cap_mib = cap;
    return c;
}

}  // namespace

TEST(the_switch_off_reads_nothing_else) {
    for (const char* v : { (const char*)nullptr, "", "0" }) {
        Env e;
        if (v) e.v[kEnv] = v;
        e.v[kPolicyEnv] = "bogus";   /* would be refused if read */
        e.v[kDeviceEnv] = "nonsense";
        Config c;
        std::string why;
        CHECK_EQ(parse_env(e, &c, &why), 0);
        CHECK(!c.on);
        CHECK_EQ(e.read.size(), (size_t)1);
        CHECK_EQ(e.read[0], std::string(kEnv));
    }
}

TEST(the_switch_on_and_refused) {
    Env e;
    e.v[kEnv] = "26000";
    Config c;
    std::string why;
    CHECK_EQ(parse_env(e, &c, &why), 1);
    CHECK(c.on);
    CHECK_EQ(c.cap_mib, (int64_t)26000);
    CHECK(!c.verify);
    CHECK_EQ(c.headroom_mib, kHeadroomDef);
    CHECK_EQ(c.deadline_us, kDeadlineDef);
    CHECK_EQ(c.backoff, kBackoffDef);
    CHECK_EQ(c.force_late, (int64_t)0);
    CHECK_EQ(c.rows, (int64_t)16);   /* RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS unset: today's 16 */
    CHECK(c.device.empty());
    for (const char* v : { "512", "-1", "26000x", " 26000", "on", "1048577", "+26000", "1023" }) {
        Env b;
        b.v[kEnv] = v;
        CHECK_EQ(parse_env(b, &c, &why), -1);
        CHECK(has(why, kEnv));
        CHECK(has(why, v));
    }
    for (const char* v : { "1024", "1048576" }) {
        Env b;
        b.v[kEnv] = v;
        CHECK_EQ(parse_env(b, &c, &why), 1);
    }
}

TEST(the_other_switches_ranges) {
    struct Row { const char* name; const char* val; int want; };
    const Row rows[] = {
        { kPolicyEnv, "verify", 1 }, { kPolicyEnv, "serve", 1 }, { kPolicyEnv, "", 1 }, { kPolicyEnv, "all", -1 },
        { kHeadroomEnv, "256", 1 }, { kHeadroomEnv, "65536", 1 }, { kHeadroomEnv, "255", -1 }, { kHeadroomEnv, "65537", -1 },
        { kDeadlineEnv, "100", 1 }, { kDeadlineEnv, "1000000", 1 }, { kDeadlineEnv, "99", -1 }, { kDeadlineEnv, "2ms", -1 },
        { kBackoffEnv, "0", 1 }, { kBackoffEnv, "65536", 1 }, { kBackoffEnv, "65537", -1 }, { kBackoffEnv, "-1", -1 },
        { kForceLateEnv, "0", 1 }, { kForceLateEnv, "7", 1 }, { kForceLateEnv, "1000001", -1 },
        { kRowsEnv, "16", 1 }, { kRowsEnv, "128", 1 }, { kRowsEnv, "512", 1 }, { kRowsEnv, "15", -1 },
        { kRowsEnv, "513", -1 }, { kRowsEnv, "0", -1 }, { kRowsEnv, "128 ", -1 }, { kRowsEnv, "-128", -1 },
        { kDeviceEnv, "0000:c3:00.0", 1 }, { kDeviceEnv, "C3:00.0", 1 }, { kDeviceEnv, "GPU-0123456789abcdef", 1 },
        { kDeviceEnv, "GPU-0123456789ABCDEF", 1 }, { kDeviceEnv, "c3:00", -1 }, { kDeviceEnv, "0000:c3:20.0", -1 },
        { kDeviceEnv, "GPU-0123", -1 }, { kDeviceEnv, "4", -1 },
    };
    for (const Row& r : rows) {
        Env e;
        e.v[kEnv] = "26000";
        e.v[r.name] = r.val;
        Config c;
        std::string why;
        const int got = parse_env(e, &c, &why);
        if (got != r.want) std::fprintf(stderr, "%s=%s: got %d want %d (%s)\n", r.name, r.val, got, r.want, why.c_str());
        CHECK_EQ(got, r.want);
        if (r.want < 0) CHECK(has(why, r.name));
    }
    Env e;
    e.v[kEnv] = "26000";
    e.v[kPolicyEnv] = "verify";
    e.v[kDeadlineEnv] = "1500";
    e.v[kBackoffEnv] = "8";
    e.v[kHeadroomEnv] = "2048";
    e.v[kForceLateEnv] = "5";
    e.v[kDeviceEnv] = "C3:00.0";
    e.v[kRowsEnv] = "256";
    Config c;
    std::string why;
    REQUIRE_EQ(parse_env(e, &c, &why), 1);
    CHECK_EQ(c.rows, (int64_t)256);
    CHECK(c.verify);
    CHECK_EQ(c.deadline_us, (int64_t)1500);
    CHECK_EQ(c.backoff, (int64_t)8);
    CHECK_EQ(c.headroom_mib, (int64_t)2048);
    CHECK_EQ(c.force_late, (int64_t)5);
    CHECK_EQ(c.device, std::string("0000:c3:00.0"));
    CHECK(!c.device_uuid);
    Env u;
    u.v[kEnv] = "26000";
    u.v[kDeviceEnv] = "GPU-0123456789ABCDEF";
    REQUIRE_EQ(parse_env(u, &c, &why), 1);
    CHECK_EQ(c.device, std::string("GPU-0123456789abcdef"));
    CHECK(c.device_uuid);
}

TEST(pci_paths) {
    std::string cx, port;
    CHECK(parse_pci_path("/sys/devices/pci0000:c0/0000:c0:01.1/0000:c1:00.0/0000:c2:00.0/0000:c3:00.0", &cx, &port));
    CHECK_EQ(cx, std::string("pci0000:c0"));
    CHECK_EQ(port, std::string("0000:c0:01.1"));
    CHECK(parse_pci_path("/sys/devices/pci0000:00/0000:00:01.1/0000:03:00.0", &cx, &port));
    CHECK_EQ(cx, std::string("pci0000:00"));
    CHECK_EQ(port, std::string("0000:00:01.1"));
    CHECK(!parse_pci_path("/sys/devices/platform/foo", &cx, &port));
    CHECK(cx.empty());
    CHECK(port.empty());
    CHECK(!parse_pci_path("", &cx, &port));
    CHECK(!parse_pci_path("/sys/devices/pci0000:c0", &cx, &port));   /* no port */
}

TEST(no_candidate_wrong_isa_integrated_and_vram) {
    const Config c = on();
    /* Four visible cards: the ranks', nothing left over. */
    std::vector<Card> four = box(28 * GiB);
    four.pop_back();
    Decision d = decide(four, kRanks, c, "");
    CHECK(!d.on);
    CHECK(has(d.why, "no visible device beyond the --tp 4"));
    /* The extra card of another arch. */
    std::vector<Card> b = box(28 * GiB);
    b[4].arch = "gfx1100";
    d = decide(b, kRanks, c, "");
    CHECK(!d.on);
    CHECK(has(d.why, "wrong ISA (gfx1100; the ranks' gfx1201)"));
    /* Integrated graphics. */
    b = box(28 * GiB);
    b[4].integrated = true;
    d = decide(b, kRanks, c, "");
    CHECK(!d.on);
    CHECK(has(d.why, "integrated"));
    /* Too little free VRAM, then exactly the cap plus the headroom. */
    const int64_t need = (c.cap_mib + c.headroom_mib) << 20;
    d = decide(box(need - 1), kRanks, c, "");
    CHECK(!d.on);
    CHECK(has(d.why, "too little free VRAM"));
    d = decide(box(need), kRanks, c, "");
    CHECK(d.on);
    CHECK_EQ(d.ord, 4);
    CHECK(!d.warn);
    /* The report: the chosen card's port and complex, and rank 3 sharing the complex on another port. */
    CHECK(has(d.topology, "PCI root port 0000:c0:01.1 (complex pci0000:c0)"));
    CHECK(has(d.topology, "shared with rank 3's card 0000:c6:00.0 (root port 0000:c0:03.1)"));
    /* An unknown root. */
    b = box(28 * GiB);
    b[4].complex.clear();
    b[4].port.clear();
    d = decide(b, kRanks, c, "");
    CHECK(d.on);
    CHECK(has(d.topology, "root unknown"));
}

TEST(a_mixed_tp_set_with_a_same_arch_leftover_warns_and_stays_off) {
    /* The production accident: the R9700 became rank 0 and an XT is left over. */
    std::vector<Card> b = box(28 * GiB);
    std::swap(b[0].name, b[4].name);
    const Decision d = decide(b, kRanks, on(), "");
    CHECK(!d.on);
    CHECK(d.warn);
    CHECK(has(d.why, "the ranks' cards mix AMD Radeon AI PRO R9700 and AMD Radeon RX 9070 XT while AMD Radeon RX 9070 XT (gfx1201) is left over"));
    CHECK(has(d.why, "list the rank cards first in ROCR_VISIBLE_DEVICES"));
}

TEST(the_selector) {
    Config c = on();
    c.device = "0000:c6:00.0";   /* rank 3's */
    Decision d = decide(box(28 * GiB), kRanks, c, c.device);
    CHECK(!d.on);
    CHECK(has(d.why, "is rank 3's card"));
    c.device = "0000:99:00.0";
    d = decide(box(28 * GiB), kRanks, c, c.device);
    CHECK(!d.on);
    CHECK(has(d.why, "names no visible device beyond the ranks'"));
    c.device = "0000:c3:00.0";
    d = decide(box(28 * GiB), kRanks, c, c.device);
    CHECK(d.on);
    CHECK_EQ(d.ord, 4);
    /* A UUID resolved through a fake KFD topology: node 1 is the R9700, node 2 rank 3's XT. */
    const std::string root = radtest::tmpdir() + "/sysfs";
    auto node = [&](int n, uint64_t uid, uint64_t gfx, uint64_t dom, uint64_t loc) {
        const std::string d1 = root + "/class/kfd/kfd/topology/nodes/" + std::to_string(n);
        ::mkdir(root.c_str(), 0755);
        for (const char* part : { "/class", "/class/kfd", "/class/kfd/kfd", "/class/kfd/kfd/topology",
                                  "/class/kfd/kfd/topology/nodes" })
            ::mkdir((root + part).c_str(), 0755);
        ::mkdir(d1.c_str(), 0755);
        FILE* f = std::fopen((d1 + "/properties").c_str(), "w");
        if (!f) return false;
        std::fprintf(f, "cpu_cores_count 0\nsimd_count 128\nunique_id %llu\ngfx_target_version %llu\ndomain %llu\n"
                        "location_id %llu\n", (unsigned long long)uid, (unsigned long long)gfx,
                     (unsigned long long)dom, (unsigned long long)loc);
        std::fclose(f);
        return true;
    };
    REQUIRE(node(0, 0, 0, 0, 0));   /* the CPU */
    REQUIRE(node(1, 0x0123456789abcdefull, 120001, 0, (0xc3u << 8) | (0 << 3) | 0));
    REQUIRE(node(2, 0xe49179ac81aef5adull, 120001, 0, (0xc6u << 8)));
    std::string bdf;
    CHECK(resolve_uuid(root, "GPU-0123456789abcdef", &bdf));
    CHECK_EQ(bdf, std::string("0000:c3:00.0"));
    CHECK(resolve_uuid(root, "GPU-E49179AC81AEF5AD", &bdf));
    CHECK_EQ(bdf, std::string("0000:c6:00.0"));
    CHECK(!resolve_uuid(root, "GPU-ffffffffffffffff", &bdf));
    CHECK_EQ(uuid_of(0xe49179ac81aef5adull), std::string("GPU-e49179ac81aef5ad"));
    Env e;
    e.v[kEnv] = "26000";
    e.v[kDeviceEnv] = "GPU-0123456789abcdef";
    std::string why;
    REQUIRE_EQ(parse_env(e, &c, &why), 1);
    CHECK_EQ(rad::aux::selector_bdf(c, root), std::string("0000:c3:00.0"));
    d = decide(box(28 * GiB), kRanks, c, rad::aux::selector_bdf(c, root));
    CHECK(d.on);
    CHECK_EQ(d.ord, 4);
    e.v[kDeviceEnv] = "GPU-E49179AC81AEF5AD";   /* rank 3's */
    REQUIRE_EQ(parse_env(e, &c, &why), 1);
    d = decide(box(28 * GiB), kRanks, c, rad::aux::selector_bdf(c, root));
    CHECK(!d.on);
    CHECK(has(d.why, "is rank 3's card"));
    e.v[kDeviceEnv] = "GPU-ffffffffffffffff";   /* no node */
    REQUIRE_EQ(parse_env(e, &c, &why), 1);
    CHECK(rad::aux::selector_bdf(c, root).empty());
    d = decide(box(28 * GiB), kRanks, c, rad::aux::selector_bdf(c, root));
    CHECK(!d.on);
    CHECK(has(d.why, "names no visible device"));
}

TEST(two_candidates_independent_root_then_free_vram_then_bdf) {
    const Config c = on();
    std::vector<Card> b = box(28 * GiB);
    /* A second spare card on a root no rank shares, with less free VRAM: it wins on the root. */
    b.push_back(card(5, kR9700, "gfx1201", "0000:e3:00.0", "pci0000:e0", "0000:e0:01.1", 27 * GiB + 512 * (1 << 20)));
    Decision d = decide(b, kRanks, c, "");
    CHECK(d.on);
    CHECK_EQ(d.ord, 5);
    CHECK(!has(d.topology, "shared"));
    /* Both on shared roots: more free VRAM wins. */
    b[5].complex = "pci0000:00";
    d = decide(b, kRanks, c, "");
    CHECK(d.on);
    CHECK_EQ(d.ord, 4);   /* 28 GiB against 27.5 */
    /* The same free VRAM: the lower BDF. */
    b[5].vram_free = b[4].vram_free;
    d = decide(b, kRanks, c, "");
    CHECK_EQ(d.ord, 4);   /* 0000:c3 before 0000:e3 */
    std::swap(b[4].bdf, b[5].bdf);
    d = decide(b, kRanks, c, "");
    CHECK_EQ(d.ord, 5);
    /* An unknown root never counts as independent. */
    b = box(28 * GiB);
    b.push_back(card(5, kR9700, "gfx1201", "0000:e3:00.0", "", "", 30 * GiB));
    d = decide(b, kRanks, c, "");
    CHECK_EQ(d.ord, 5);   /* more free VRAM; neither root independent (c0 is shared, e3's unknown) */
    /* Every candidate failing: each listed with its reason. */
    b = box(1 * GiB);
    b.push_back(card(5, kR9700, "gfx1100", "0000:e3:00.0", "", "", 30 * GiB));
    d = decide(b, kRanks, c, "");
    CHECK(!d.on);
    CHECK(has(d.why, "device 4 (0000:c3:00.0, AMD Radeon AI PRO R9700): too little free VRAM"));
    CHECK(has(d.why, "device 5 (0000:e3:00.0, AMD Radeon AI PRO R9700): wrong ISA"));
}

TEST(strip_arch_and_bdf_forms) {
    CHECK_EQ(strip_arch("gfx1201:sramecc-:xnack-"), std::string("gfx1201"));
    CHECK_EQ(strip_arch("gfx1201"), std::string("gfx1201"));
    std::string n;
    CHECK(parse_bdf("0000:C3:00.0", &n));
    CHECK_EQ(n, std::string("0000:c3:00.0"));
    CHECK(parse_bdf("c3:1f.7", &n));
    CHECK_EQ(n, std::string("0000:c3:1f.7"));
    CHECK(!parse_bdf("c3:1f.8", &n));
    CHECK(!parse_bdf("0000:c3:00", &n));
    CHECK(!parse_bdf("0000:g3:00.0", &n));
}

RAD_TEST_MAIN()
