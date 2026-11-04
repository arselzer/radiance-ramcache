/* aux_share_test.cpp -- the aux expert worker's card sharing, pure parts (tools/ds41-host-experts/ds41_aux_share.h;
 * work/aux-dyn/SPEC.md part A, §7.1), and the new switches (ds41_aux.h parse_dynamic), on a fake sysfs and procfs
 * under a temp root, a fake clock and temp files. No card, no core.
 *  - the probe: the card's gpu_id by its PCI address, mem_info, clients by vram_<gpu_id> (this process left out, a
 *    process under the minimum left out, an unreadable comm as "?", a KFD entry with no /proc entry a client only
 *    after 10 s), appeared / gone, a missing entry of this process (another pid namespace) leaving the clients
 *    unknown, the always policy reading the room only;
 *  - judge: exclusive, allow (the regex, the cap), the floor while serving, the room while released (the free
 *    counter and the driver's own figure), unknown;
 *  - the state machine: a client the policy refuses releases at once, the floor and an unknown probe only after N
 *    polls; the free clock reset by one busy poll; the reacquire after R s; the backoff doubled by a policy release
 *    within 600 s of a counted acquire (not the first after a released start), reset after 1800 s serving, doubled
 *    by a refused acquire; manual off / on / auto in every state (A.3); loading aborted; dead stays dead; the
 *    startup watch never acquires;
 *  - the control file (words, case, garbage once per content, missing: auto, empty: no change, rewrites seen by
 *    inode, size and mtime) and the holders (a shared flock held: live; given back: gone; dot files and
 *    directories ignored);
 *  - the state line and /proc/<pid>/stat's start time (a comm with spaces and parentheses); write_atomic;
 *  - the switches: defaults, every range's edges, a sub-switch without its master refused naming both, the regex
 *    refused, the control path's checks, nothing read with the worker off. */
#include "rad_test.h"

#include "../tools/ds41-host-experts/ds41_aux_share.h"

#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace sh = ds41aux::share;

namespace {

void put(const std::string& path, const std::string& text) {
    const size_t k = path.rfind('/');
    std::string dir = path.substr(0, k);
    std::string acc;
    for (size_t i = 1; i <= dir.size(); ++i)
        if (i == dir.size() || dir[i] == '/') {
            acc = dir.substr(0, i);
            ::mkdir(acc.c_str(), 0777);
        }
    std::ofstream f(path, std::ios::trunc);
    f << text;
}

bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

/* A fake tree: the CPU node, the extra card at 0000:c3:00.0 (gpu_id 50005), its mem_info, this process's KFD entry. */
struct Tree {
    std::string root, sys, proc;
    long self = 4242;
    explicit Tree(const std::string& name) {
        root = radtest::tmp_path(name);
        sys = root + "/sys";
        proc = root + "/proc";
        put(sys + "/class/kfd/kfd/topology/nodes/0/properties", "cpu_cores_count 48\ngfx_target_version 0\n");
        put(sys + "/class/kfd/kfd/topology/nodes/0/gpu_id", "0\n");
        /* location_id: bus << 8 | device << 3 | function */
        put(sys + "/class/kfd/kfd/topology/nodes/5/properties",
            "gfx_target_version 120001\ndomain 0\nlocation_id " + std::to_string(0xc3 << 8) + "\nunique_id 170\n");
        put(sys + "/class/kfd/kfd/topology/nodes/5/gpu_id", "50005\n");
        put(sys + "/class/kfd/kfd/topology/nodes/1/properties",
            "gfx_target_version 120001\ndomain 0\nlocation_id " + std::to_string(0x03 << 8) + "\nunique_id 171\n");
        put(sys + "/class/kfd/kfd/topology/nodes/1/gpu_id", "11111\n");
        put(sys + "/bus/pci/devices/0000:c3:00.0/mem_info_vram_total", std::to_string(32LL << 30) + "\n");
        used(5LL << 30);
        client(self, "radiance", 26LL << 30);
    }
    void used(int64_t b) { put(sys + "/bus/pci/devices/0000:c3:00.0/mem_info_vram_used", std::to_string(b) + "\n"); }
    void client(long pid, const std::string& comm, int64_t bytes, bool proc_entry = true) {
        put(sys + "/class/kfd/kfd/proc/" + std::to_string(pid) + "/vram_50005", std::to_string(bytes) + "\n");
        put(sys + "/class/kfd/kfd/proc/" + std::to_string(pid) + "/vram_11111", "999999999\n");
        if (proc_entry) put(proc + "/" + std::to_string(pid) + "/comm", comm + "\n");
    }
    void gone(long pid) {
        std::string cmd = "rm -rf '" + sys + "/class/kfd/kfd/proc/" + std::to_string(pid) + "' '" + proc + "/" +
                          std::to_string(pid) + "'";
        REQUIRE(std::system(cmd.c_str()) == 0);
    }
};

ds41aux::Config cfg_allow() {
    ds41aux::Config c;
    c.on = true;
    c.cap_mib = 26000;
    c.headroom_mib = 1024;
    c.share = ds41aux::kShareAllow;
    return c;
}

}  // namespace

/* ---- the probe ---- */

TEST(probe_finds_the_card_and_lists_its_clients) {
    Tree t("probe1");
    sh::Probe p;
    std::string why;
    REQUIRE_EQ(p.init(t.sys, t.proc, "0000:c3:00.0", t.self, true, &why), 0);
    CHECK_EQ(p.gpu_id(), (uint32_t)50005);
    CHECK(p.self_ok());
    t.client(100, "whisper-server", 4404019200LL);
    t.client(101, "observer", 512 * 1024);          /* under 1 MiB: not a client */
    t.client(102, "", 2LL << 20);                    /* an unreadable comm */
    sh::Poll a = p.poll(1000, 1LL << 20);
    REQUIRE(a.ok_room && a.ok_clients);
    CHECK_EQ(a.total, 32LL << 30);
    CHECK_EQ(a.free_b, 27LL << 30);
    CHECK_EQ(a.self, 26LL << 30);
    REQUIRE_EQ(a.clients.size(), (size_t)2);         /* self and the observer left out */
    CHECK_EQ(a.clients[0].pid, 100);
    CHECK_EQ(a.clients[0].comm, std::string("whisper-server"));
    CHECK_EQ(a.clients[1].comm, std::string("?"));
    CHECK_EQ(a.appeared.size(), (size_t)2);
    /* min 0: the observer too; 16 MiB: whisper alone */
    sh::Poll a0 = p.poll(1001, 0);
    CHECK_EQ(a0.clients.size(), (size_t)3);
    sh::Poll a16 = p.poll(1002, 16LL << 20);
    CHECK_EQ(a16.clients.size(), (size_t)1);
    /* gone and appeared */
    sh::Poll b = p.poll(1003, 1LL << 20);
    CHECK_EQ(b.appeared.size(), (size_t)1);   /* pid 102 again, beside the 16 MiB poll's whisper */
    t.gone(102);
    sh::Poll c = p.poll(1004, 1LL << 20);
    CHECK_EQ(c.clients.size(), (size_t)1);
    REQUIRE_EQ(c.gone.size(), (size_t)1);
    CHECK_EQ(c.gone[0].pid, 102);
}

TEST(probe_a_kfd_entry_without_a_proc_entry_counts_after_ten_seconds) {
    Tree t("probe2");
    sh::Probe p;
    std::string why;
    REQUIRE_EQ(p.init(t.sys, t.proc, "0000:c3:00.0", t.self, true, &why), 0);
    t.client(300, "hidden", 8LL << 30, /*proc_entry=*/false);
    CHECK_EQ(p.poll(1000, 1LL << 20).clients.size(), (size_t)0);
    CHECK_EQ(p.poll(10999, 1LL << 20).clients.size(), (size_t)0);
    sh::Poll q = p.poll(11000, 1LL << 20);
    REQUIRE_EQ(q.clients.size(), (size_t)1);
    CHECK(q.clients[0].no_proc);
    CHECK_EQ(q.clients[0].comm, std::string("?"));
    /* an entry that went within the 10 s (an exited process) never counted, and its clock is forgotten */
    t.gone(300);
    t.client(301, "x", 8LL << 30, false);
    CHECK_EQ(p.poll(12000, 1LL << 20).clients.size(), (size_t)0);
    t.gone(301);
    CHECK_EQ(p.poll(13000, 1LL << 20).clients.size(), (size_t)0);
    t.client(301, "x", 8LL << 30, false);
    CHECK_EQ(p.poll(22500, 1LL << 20).clients.size(), (size_t)0);   /* a new clock from 22500 */
}

TEST(probe_without_its_own_kfd_entry_leaves_the_clients_unknown) {
    Tree t("probe3");
    sh::Probe p;
    std::string why;
    REQUIRE_EQ(p.init(t.sys, t.proc, "0000:c3:00.0", /*self=*/777, true, &why), 0);
    CHECK(!p.self_ok());
    CHECK(contains(p.why_self(), "another pid namespace"));
    sh::Poll q = p.poll(1000, 1LL << 20);
    CHECK(q.ok_room);
    CHECK(!q.ok_clients);
    const sh::Verdict v = sh::judge(cfg_allow(), sh::Allow{}, q, true, -1);
    CHECK(v.unknown_clients);
    /* the always policy reads the room only */
    sh::Probe pa;
    REQUIRE_EQ(pa.init(t.sys, t.proc, "0000:c3:00.0", 777, false, &why), 0);
    CHECK(pa.self_ok());
    sh::Poll qa = pa.poll(1000, 1LL << 20);
    CHECK(qa.ok_room && !qa.ok_clients);
    /* no card counters: init says so, the room unknown */
    sh::Probe pb;
    CHECK(pb.init(t.sys, t.proc, "0000:c4:00.0", t.self, true, &why) < 0);
    CHECK(!pb.poll(1000, 0).ok_room);
}

/* ---- the verdict ---- */

TEST(judge_by_policy) {
    sh::Allow allow;
    REQUIRE(allow.set("^whisper"));
    sh::Poll p;
    p.ok_room = p.ok_clients = true;
    p.total = 32LL << 30;
    p.free_b = 2LL << 30;
    p.clients = { { 100, "whisper-server", 4404019200LL, false } };
    ds41aux::Config c = cfg_allow();
    sh::Verdict v = sh::judge(c, allow, p, /*serving=*/true, -1);
    CHECK(!v.foreign && !v.cap && !v.floor && !v.unknown_clients);
    c.share = ds41aux::kShareExclusive;
    v = sh::judge(c, allow, p, true, -1);
    CHECK(v.foreign);
    CHECK(contains(v.why, "whisper-server (pid 100) 4200 MiB"));
    c = cfg_allow();
    p.clients.push_back({ 200, "python3", 12LL << 30, false });
    v = sh::judge(c, allow, p, true, -1);
    CHECK(v.foreign);
    CHECK(contains(v.why, "python3"));
    p.clients.pop_back();
    c.foreign_mib = 4000;   /* whisper's 4200 MiB above the cap */
    v = sh::judge(c, allow, p, true, -1);
    CHECK(!v.foreign && v.cap);
    c = cfg_allow();
    /* the floor while serving */
    p.free_b = 1000LL << 20;
    v = sh::judge(c, allow, p, true, -1);
    CHECK(v.floor);
    c.floor_mib = 0;
    CHECK(!sh::judge(c, allow, p, true, -1).floor);
    c = cfg_allow();
    /* the room while released: free >= cap + headroom + 512 MiB, the driver's figure >= cap + headroom */
    p.free_b = (26000LL + 1024 + 512) << 20;
    v = sh::judge(c, allow, p, false, -1);
    CHECK(!v.no_room && v.free_for_auto() && v.room());
    p.free_b -= 1;
    CHECK(sh::judge(c, allow, p, false, -1).no_room);
    p.free_b += 1;
    CHECK(sh::judge(c, allow, p, false, (26000LL + 1024) << 20).room());
    v = sh::judge(c, allow, p, false, ((26000LL + 1024) << 20) - 1);
    CHECK(v.no_room);
    CHECK(contains(v.why, "the driver would give"));
    /* unknown */
    p.ok_clients = false;
    p.why = "x";
    v = sh::judge(c, allow, p, false, -1);
    CHECK(v.unknown_clients && !v.free_for_auto() && v.room());
    p.ok_room = false;
    v = sh::judge(c, allow, p, false, -1);
    CHECK(v.unknown_room && !v.room());
    /* always: no client rule, the floor with a controller */
    c.share = ds41aux::kShareAlways;
    c.control = "/tmp/x";
    p.ok_room = true;
    p.ok_clients = false;
    p.free_b = 100LL << 20;
    v = sh::judge(c, allow, p, true, -1);
    CHECK(!v.unknown_clients && !v.foreign && v.floor);
}

/* ---- the state machine ---- */

namespace {
sh::Verdict ok_v() { return sh::Verdict{}; }
sh::Verdict foreign_v() { sh::Verdict v; v.foreign = true; v.why = "f"; return v; }
sh::Verdict floor_v() { sh::Verdict v; v.floor = true; v.why = "fl"; return v; }
sh::Verdict unknown_v() { sh::Verdict v; v.unknown_clients = true; v.why = "u"; return v; }
sh::Verdict noroom_v() { sh::Verdict v; v.no_room = true; v.why = "nr"; return v; }
}  // namespace

TEST(fsm_release_rules) {
    using sh::Act;
    using sh::Mode;
    sh::Fsm f(/*polls=*/2, /*reacquire_ms=*/60000);
    f.start(sh::St::kServing, 0, false);
    CHECK(f.step(100, foreign_v(), Mode::kAuto).act == Act::kRelease);   /* at once */
    CHECK(f.state() == sh::St::kDraining);
    CHECK(f.step(200, foreign_v(), Mode::kAuto).act == Act::kNone);      /* draining continues */
    f.done_release(300, true);
    CHECK(f.state() == sh::St::kReleased);
    for (const auto& v : { floor_v(), unknown_v() }) {
        sh::Fsm g(2, 60000);
        g.start(sh::St::kServing, 0, false);
        CHECK(g.step(100, v, Mode::kAuto).act == Act::kNone);   /* N - 1 polls */
        CHECK(g.step(200, ok_v(), Mode::kAuto).act == Act::kNone);   /* a good poll restarts the count */
        CHECK(g.step(300, v, Mode::kAuto).act == Act::kNone);
        const sh::Step s = g.step(400, v, Mode::kAuto);
        CHECK(s.act == Act::kRelease);
        CHECK(s.why == (v.floor ? sh::kWhyFloor : sh::kWhyUnknown));
    }
    /* manual on: only the floor */
    sh::Fsm h(1, 60000);
    h.start(sh::St::kServing, 0, false);
    CHECK(h.step(100, foreign_v(), Mode::kOn).act == Act::kNone);
    CHECK(h.step(200, unknown_v(), Mode::kOn).act == Act::kNone);
    CHECK(h.step(300, floor_v(), Mode::kOn).act == Act::kRelease);
    /* manual off: at once, whatever the verdict */
    sh::Fsm k(5, 60000);
    k.start(sh::St::kServing, 0, false);
    const sh::Step s = k.step(100, ok_v(), Mode::kOff);
    CHECK(s.act == Act::kRelease && s.why == sh::kWhyManual);
}

TEST(fsm_reacquire_clock_and_backoff) {
    using sh::Act;
    using sh::Mode;
    sh::Fsm f(2, 60000);
    f.start(sh::St::kReleased, 0, /*released_at_start=*/true);
    CHECK(f.step(0, ok_v(), Mode::kAuto).act == Act::kNone);
    CHECK(f.step(30000, ok_v(), Mode::kAuto).act == Act::kNone);
    CHECK(f.step(30100, noroom_v(), Mode::kAuto).act == Act::kNone);   /* restarts the clock */
    CHECK(f.step(30200, ok_v(), Mode::kAuto).act == Act::kNone);
    CHECK(f.step(90100, ok_v(), Mode::kAuto).act == Act::kNone);
    CHECK(f.step(90200, ok_v(), Mode::kAuto).act == Act::kAcquire);
    CHECK(f.state() == sh::St::kLoading);
    f.done_acquire(91000, 1);   /* the first after a released start: not counted for the backoff */
    CHECK(f.step(100000, foreign_v(), Mode::kAuto).act == Act::kRelease);
    CHECK_EQ(f.backoff(), 1);
    f.done_release(100100, true);
    CHECK(f.step(100200, ok_v(), Mode::kAuto).act == Act::kNone);
    CHECK(f.step(160200, ok_v(), Mode::kAuto).act == Act::kAcquire);
    f.done_acquire(160300, 1);   /* counted */
    CHECK(f.step(200000, foreign_v(), Mode::kAuto).act == Act::kRelease);
    CHECK_EQ(f.backoff(), 2);   /* a policy release within 600 s of a counted acquire */
    f.done_release(200100, true);
    CHECK_EQ(f.reacquire_ms(), 120000);
    CHECK(f.step(200200, ok_v(), Mode::kAuto).act == Act::kNone);
    CHECK(f.step(320100, ok_v(), Mode::kAuto).act == Act::kNone);
    CHECK(f.step(320200, ok_v(), Mode::kAuto).act == Act::kAcquire);
    f.done_acquire(320300, 0);   /* refused: doubled */
    CHECK_EQ(f.backoff(), 4);
    CHECK(f.state() == sh::St::kReleased);
    CHECK(f.step(320400, ok_v(), Mode::kAuto).act == Act::kNone);
    CHECK(f.step(560400, ok_v(), Mode::kAuto).act == Act::kAcquire);
    f.done_acquire(560500, 1);
    /* a manual release leaves the backoff; serving 1800 s resets it */
    sh::Fsm g = f;
    CHECK(g.step(560600, ok_v(), Mode::kOff).act == Act::kRelease);
    CHECK_EQ(g.backoff(), 4);
    CHECK(f.step(560500 + 1800000, ok_v(), Mode::kAuto).act == Act::kNone);
    CHECK_EQ(f.backoff(), 1);
    /* up to 16 */
    sh::Fsm m(1, 1000);
    m.start(sh::St::kReleased, 0, false);
    for (int i = 0; i < 8; ++i) {
        m.done_acquire(0, 0);
    }
    CHECK_EQ(m.backoff(), 16);
}

TEST(fsm_manual_modes_in_every_state) {
    using sh::Act;
    using sh::Mode;
    /* released: off stays, on acquires at the first poll with room, auto waits for R s */
    sh::Fsm r(2, 60000);
    r.start(sh::St::kReleased, 0, false);
    CHECK(r.step(10, ok_v(), Mode::kOff).act == Act::kNone);
    CHECK(r.step(20, noroom_v(), Mode::kOn).act == Act::kNone);
    sh::Verdict uk = unknown_v();
    CHECK(r.step(30, uk, Mode::kOn).act == Act::kAcquire);   /* manual on ignores the clients: the room is known */
    /* loading: off aborts, on continues, auto aborts on foreign at once and the floor after N */
    sh::Fsm l(2, 0);
    l.start(sh::St::kReleased, 0, false);
    REQUIRE(l.step(0, ok_v(), Mode::kAuto).act == Act::kAcquire);
    CHECK(l.state() == sh::St::kLoading);
    CHECK(l.step(10, foreign_v(), Mode::kOn).act == Act::kNone);
    CHECK(l.step(20, floor_v(), Mode::kOn).act == Act::kNone);
    CHECK(l.step(30, floor_v(), Mode::kAuto).act == Act::kNone);
    CHECK(l.step(40, floor_v(), Mode::kAuto).act == Act::kRelease);
    sh::Fsm l2(2, 0);
    l2.start(sh::St::kReleased, 0, false);
    REQUIRE(l2.step(0, ok_v(), Mode::kAuto).act == Act::kAcquire);
    CHECK(l2.step(10, ok_v(), Mode::kOff).act == Act::kRelease);
    /* dead stays */
    sh::Fsm d(1, 0);
    d.start(sh::St::kServing, 0, false);
    REQUIRE(d.step(0, foreign_v(), Mode::kAuto).act == Act::kRelease);
    d.done_release(1, false);
    CHECK(d.state() == sh::St::kDead);
    for (Mode m : { Mode::kAuto, Mode::kOn, Mode::kOff }) CHECK(d.step(100, ok_v(), m).act == Act::kNone);
    /* starting (the startup watch): releases, never acquires */
    sh::Fsm s(1, 0);
    s.start(sh::St::kStarting, 0, false);
    CHECK(s.step(10, ok_v(), Mode::kAuto).act == Act::kNone);
    CHECK(s.step(20, ok_v(), Mode::kOn).act == Act::kNone);
    CHECK(s.step(30, ok_v(), Mode::kOff).act == Act::kRelease);
}

/* ---- manual control ---- */

TEST(control_file_words_garbage_and_changes) {
    sh::Mode m;
    CHECK_EQ(sh::parse_mode("off\n", &m), 1);
    CHECK(m == sh::Mode::kOff);
    CHECK_EQ(sh::parse_mode("  ON  trailing", &m), 1);
    CHECK(m == sh::Mode::kOn);
    CHECK_EQ(sh::parse_mode("Auto", &m), 1);
    CHECK(m == sh::Mode::kAuto);
    CHECK_EQ(sh::parse_mode("", &m), 0);
    CHECK_EQ(sh::parse_mode(" \n", &m), 0);
    CHECK_EQ(sh::parse_mode("offf", &m), -1);
    const std::string p = radtest::tmp_path("ctl1");
    sh::ControlFile cf(p);
    std::string g;
    CHECK(!cf.poll(&m, &g));   /* missing: auto */
    CHECK(m == sh::Mode::kAuto);
    REQUIRE(sh::write_atomic(p, "off\n"));
    CHECK(cf.poll(&m, &g));
    CHECK(m == sh::Mode::kOff);
    CHECK(!cf.poll(&m, &g));   /* no change */
    { std::ofstream f(p, std::ios::trunc); }   /* `echo off > f` caught between its truncate and its write */
    CHECK(!cf.poll(&m, &g));
    CHECK(m == sh::Mode::kOff);
    REQUIRE(sh::write_atomic(p, "bogus\n"));
    CHECK(!cf.poll(&m, &g));
    CHECK_EQ(g, std::string("bogus"));
    CHECK(m == sh::Mode::kOff);
    REQUIRE(sh::write_atomic(p, "bogus\n"));   /* the same garbage: warned once */
    CHECK(!cf.poll(&m, &g));
    CHECK(g.empty());
    REQUIRE(sh::write_atomic(p, "on\n"));
    CHECK(cf.poll(&m, &g));
    CHECK(m == sh::Mode::kOn);
    REQUIRE(sh::write_atomic(p, "on\n"));      /* rewritten, same word: seen, no change */
    CHECK(!cf.poll(&m, &g));
    ::unlink(p.c_str());
    CHECK(cf.poll(&m, &g));   /* removed: auto */
    CHECK(m == sh::Mode::kAuto);
}

TEST(holders_by_flock) {
    const std::string d = radtest::tmp_path("ctl2.d");
    ::mkdir(d.c_str(), 0777);
    CHECK(sh::live_holders(d).empty());
    put(d + "/job-1", "");
    put(d + "/.tmp-2", "");
    ::mkdir((d + "/dir").c_str(), 0777);
    CHECK(sh::live_holders(d).empty());   /* no lock: a crashed holder's file */
    const int fd = ::open((d + "/job-1").c_str(), O_RDONLY);
    REQUIRE(fd >= 0);
    REQUIRE(::flock(fd, LOCK_SH) == 0);
    const int fd2 = ::open((d + "/.tmp-2").c_str(), O_RDONLY);
    REQUIRE(::flock(fd2, LOCK_SH) == 0);   /* a dot file: not yet a holder */
    std::vector<std::string> h = sh::live_holders(d);
    REQUIRE_EQ(h.size(), (size_t)1);
    CHECK_EQ(h[0], std::string("job-1"));
    CHECK_EQ(sh::live_holders(d).size(), (size_t)1);   /* the probe gave its own lock back */
    ::close(fd);
    ::close(fd2);
    CHECK(sh::live_holders(d).empty());
    CHECK(sh::live_holders(d + "/none").empty());
}

TEST(state_line_and_start_time) {
    const std::string l = sh::state_line("dead held 26112", 1762245000123LL, 77, 123456, sh::Mode::kOff,
                                         "holder job-1", "a lane\nfaulted");
    CHECK_EQ(l, std::string("dead held 26112 1762245000123 pid 77 123456 mode off (holder job-1) a lane faulted\n"));
    const std::string r = radtest::tmp_path("procst");
    put(r + "/77/stat", "77 (we (ird) x) S 1 77 77 0 -1 4194560 100 0 0 0 5 3 0 0 20 0 9 0 98765 1000 200\n");
    CHECK_EQ(sh::proc_starttime(r, 77), 98765ULL);
    CHECK_EQ(sh::proc_starttime(r, 78), 0ULL);
    const std::string p = radtest::tmp_path("statefile");
    REQUIRE(sh::write_atomic(p, "serving 1 pid 2 3 mode auto (default)\n"));
    std::string t;
    REQUIRE(sh::read_file(p, &t));
    CHECK_EQ(t, std::string("serving 1 pid 2 3 mode auto (default)\n"));
}

/* ---- the switches ---- */

namespace {
struct Env {
    std::map<std::string, std::string> v;
    std::set<std::string> asked;
    const char* operator()(const char* k) {
        asked.insert(k);
        auto it = v.find(k);
        return it == v.end() ? nullptr : it->second.c_str();
    }
};
int parse(Env& e, ds41aux::Config* c, std::string* why) {
    return ds41aux::parse([&](const char* k) { return e(k); }, c, why);
}
}  // namespace

TEST(switches_defaults_ranges_and_masters) {
    ds41aux::Config c;
    std::string why;
    Env off;
    CHECK_EQ(parse(off, &c, &why), 0);
    CHECK_EQ(off.asked.size(), (size_t)1);   /* the worker off: nothing else read */
    Env e;
    e.v[ds41aux::kEnv] = "26000";
    REQUIRE_EQ(parse(e, &c, &why), 1);
    CHECK(c.share == ds41aux::kShareAlways && c.control.empty() && !c.online);
    CHECK(!c.controller() && !c.control_thread());
    CHECK_EQ(c.min_mib, 1);
    CHECK_EQ(c.poll_ms, 100);
    CHECK_EQ(c.release_polls, 2);
    CHECK_EQ(c.floor_mib, 1024);
    CHECK_EQ(c.allow, std::string("^whisper"));
    CHECK_EQ(c.rate, 200);
    CHECK_EQ(c.spare, 16);
    /* the policy words */
    for (const char* w : { "always", "exclusive", "allow" }) {
        Env x = e;
        x.v[ds41aux::kShareEnv] = w;
        REQUIRE_EQ(parse(x, &c, &why), 1);
    }
    CHECK(c.share == ds41aux::kShareAllow && c.controller() && c.control_thread());
    {
        Env x = e;
        x.v[ds41aux::kShareEnv] = "sometimes";
        CHECK_EQ(parse(x, &c, &why), -1);
        CHECK(contains(why, ds41aux::kShareEnv));
    }
    /* a sub-switch without its master refuses naming both */
    struct Sub { const char* name; const char* val; const char* master; };
    const Sub subs[] = { { ds41aux::kShareAllowEnv, "^w", "=allow" },
                         { ds41aux::kShareForeignEnv, "100", "=allow" },
                         { ds41aux::kShareMinEnv, "16", "=exclusive or allow" },
                         { ds41aux::kShareFloorEnv, "100", ds41aux::kControlEnv },
                         { ds41aux::kSharePollEnv, "100", ds41aux::kControlEnv },
                         { ds41aux::kShareRelPollsEnv, "3", ds41aux::kControlEnv },
                         { ds41aux::kShareReacqEnv, "0", ds41aux::kControlEnv },
                         { ds41aux::kShareFillEnv, "1000", ds41aux::kControlEnv },
                         { ds41aux::kOnlineRateEnv, "100", ds41aux::kOnlineEnv },
                         { ds41aux::kOnlineSpareEnv, "8", ds41aux::kOnlineEnv },
                         { ds41aux::kOnlineDryEnv, "1", ds41aux::kOnlineEnv } };
    for (const Sub& s : subs) {
        Env x = e;
        x.v[s.name] = s.val;
        CHECK_EQ(parse(x, &c, &why), -1);
        CHECK(contains(why, s.name) && contains(why, s.master));
    }
    /* exclusive is not allow's master */
    {
        Env x = e;
        x.v[ds41aux::kShareEnv] = "exclusive";
        x.v[ds41aux::kShareForeignEnv] = "100";
        CHECK_EQ(parse(x, &c, &why), -1);
        x.v.erase(ds41aux::kShareForeignEnv);
        x.v[ds41aux::kShareMinEnv] = "16";
        CHECK_EQ(parse(x, &c, &why), 1);
        CHECK_EQ(c.min_mib, 16);
    }
    /* range edges with their masters on */
    struct Edge { const char* name; int64_t lo, hi; };
    const Edge edges[] = { { ds41aux::kShareForeignEnv, 0, 65536 }, { ds41aux::kShareMinEnv, 0, 65536 },
                           { ds41aux::kShareFloorEnv, 0, 65536 },   { ds41aux::kSharePollEnv, 50, 5000 },
                           { ds41aux::kShareRelPollsEnv, 1, 100 },  { ds41aux::kShareReacqEnv, 0, 86400 },
                           { ds41aux::kShareFillEnv, 100, 20000 },  { ds41aux::kOnlineRateEnv, 1, 1600 },
                           { ds41aux::kOnlineMaxEnv, 1, 256 },      { ds41aux::kOnlineEveryEnv, 1, 64 },
                           { ds41aux::kOnlineHalfEnv, 16, 65536 },  { ds41aux::kOnlinePriorEnv, 0, 65536 },
                           { ds41aux::kOnlineMarginEnv, 100, 400 }, { ds41aux::kOnlineSpareEnv, 1, 256 },
                           { ds41aux::kOnlineMbpsEnv, 100, 20000 }, { ds41aux::kOnlineCheckEnv, 0, 65536 } };
    Env m = e;
    m.v[ds41aux::kShareEnv] = "allow";
    m.v[ds41aux::kOnlineEnv] = "1";
    for (const Edge& g : edges) {
        for (int64_t v : { g.lo, g.hi }) {
            Env x = m;
            x.v[g.name] = std::to_string(v);
            CHECK_EQ(parse(x, &c, &why), 1);
        }
        for (int64_t v : { g.lo - 1, g.hi + 1 }) {
            if (v < 0) continue;
            Env x = m;
            x.v[g.name] = std::to_string(v);
            CHECK_EQ(parse(x, &c, &why), -1);
            CHECK(contains(why, g.name));
        }
        Env x = m;
        x.v[g.name] = "12x";
        CHECK_EQ(parse(x, &c, &why), -1);
    }
    {
        Env x = m;
        x.v[ds41aux::kShareAllowEnv] = "(unclosed";
        CHECK_EQ(parse(x, &c, &why), -1);
        CHECK(contains(why, "regular expression"));
        x.v[ds41aux::kShareAllowEnv] = "^(whisper|ollama)";
        CHECK_EQ(parse(x, &c, &why), 1);
        CHECK_EQ(c.allow, std::string("^(whisper|ollama)"));
        x.v[ds41aux::kOnlineEnv] = "yes";
        CHECK_EQ(parse(x, &c, &why), -1);
        x.v[ds41aux::kOnlineEnv] = "0";
        x.v.erase(ds41aux::kOnlineDryEnv);
        CHECK_EQ(parse(x, &c, &why), 1);
        CHECK(!c.online);
    }
}

TEST(switches_the_control_path) {
    ds41aux::Config c;
    std::string why;
    Env e;
    e.v[ds41aux::kEnv] = "26000";
    const std::string dir = radtest::tmp_path("ctldir");
    ::mkdir(dir.c_str(), 0777);
    e.v[ds41aux::kControlEnv] = dir + "/aux-ctl";
    REQUIRE_EQ(parse(e, &c, &why), 1);
    CHECK(c.controller() && c.share == ds41aux::kShareAlways);
    e.v[ds41aux::kControlEnv] = dir;   /* a directory */
    CHECK_EQ(parse(e, &c, &why), -1);
    e.v[ds41aux::kControlEnv] = dir + "/none/aux-ctl";
    CHECK_EQ(parse(e, &c, &why), -1);
    CHECK(contains(why, "does not exist"));
    e.v[ds41aux::kControlEnv] = dir + "/";
    CHECK_EQ(parse(e, &c, &why), -1);
    /* a controller makes the timing switches legal under always */
    e.v[ds41aux::kControlEnv] = dir + "/aux-ctl";
    e.v[ds41aux::kSharePollEnv] = "250";
    e.v[ds41aux::kShareReacqEnv] = "0";
    REQUIRE_EQ(parse(e, &c, &why), 1);
    CHECK_EQ(c.poll_ms, 250);
    CHECK_EQ(c.reacquire_s, 0);
    /* online alone: a control thread, no controller */
    Env o;
    o.v[ds41aux::kEnv] = "26000";
    o.v[ds41aux::kOnlineEnv] = "1";
    REQUIRE_EQ(parse(o, &c, &why), 1);
    CHECK(!c.controller() && c.control_thread());
}

RAD_TEST_MAIN()
