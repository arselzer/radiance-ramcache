/* Included by r4d_selftest.cpp after its collective helpers. Test only: no kernel/ABI changes.
 * The actual Engram bf16 row is 6400, assembled [token][rank][column], never reduced. */
struct AgRowsRank {
    const RadKernelInfo* row = nullptr;
    ArSeedSeq seed = nullptr;
    StBarrier* barrier = nullptr;
    std::vector<AgRowsRank>* all = nullptr;
    int rank = 0, dev = 0, cap = 1024;
    bool pitched = false;
    unsigned int handshake = 0, parity = 0;
    void *whole = nullptr, *sliced = nullptr;
    int rc = RAD_OK, setup_rc = RAD_E_STATE, idle_rc = RAD_E_STATE;
    int seed_rc = RAD_E_STATE, capture_rc = RAD_E_STATE;
    size_t bad = 0;
};
static uint16_t ag_rows_bits(int rank, int64_t row, int64_t col, unsigned epoch) {
    const uint16_t special[] = { 0, 0x8000, 0x7F80, 0xFF80, 1, 0x8001, 0x7F81, 0x7FC1 };
    if (col % 257 < 8) {
        uint16_t v = special[col % 257];
        if (col % 257 >= 6) v += (uint16_t)(rank + epoch % 31); // distinct NaN payloads
        return v;
    }
    const uint32_t x = (uint32_t)col * 40503u + (uint32_t)row * 23063u +
                       (uint32_t)rank * 32749u + epoch * 8191u;
    return (uint16_t)(x ^ (x >> 16));  // opaque bits: no bf16 conversion may canonicalize these
}
static void ag_rows_worker(AgRowsRank* t) {
    constexpr int64_t W = 6400, C = 4096, r0 = 13, rows = C + 2 * r0, guard = 128;
    const int64_t sp = W + (t->pitched ? 64 : 0), dp = 4 * W + (t->pitched ? 64 : 0);
    const size_t ni = (size_t)(2 * guard + rows * sp), no = (size_t)(2 * guard + rows * dp);
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    void *input = nullptr, *full = nullptr, *part = nullptr;
    t->rc = st_set_device(t->dev) == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    if (t->rc == RAD_OK && hipStreamCreate(&stream) != hipSuccess) t->rc = RAD_E_DEVICE;
    const RadParam whole[] = { PI("world_size", 4), PI("numel", C * W), PS("dtype", "bf16"), PI("row", W) };
    const RadParam sliced[] = { PI("world_size", 4), PI("numel", (int64_t)t->cap * W),
                               PS("dtype", "bf16"), PI("row", W) };
    if (t->rc == RAD_OK) t->rc = t->row->init(whole, 4, t->rank, 4, &t->whole);
    if (t->rc == RAD_OK) t->rc = t->row->init(sliced, 4, t->rank, 4, &t->sliced);
    if (t->rc == RAD_OK && (hipMalloc(&input, ni * 2) != hipSuccess || hipMalloc(&full, no * 2) != hipSuccess ||
                            hipMalloc(&part, no * 2) != hipSuccess)) t->rc = RAD_E_NOMEM;
    if (t->rc == RAD_OK) t->rc = t->seed(t->whole, t->handshake, t->parity);
    if (t->rc == RAD_OK) t->rc = t->seed(t->sliced, t->handshake, t->parity);
    t->setup_rc = t->rc;
    t->barrier->wait();
    bool ready = true;
    for (const auto& r : *t->all) ready = ready && r.setup_rc == RAD_OK;
    if (ready) {
        std::vector<uint16_t> host(ni), gold(no), got(no);
        auto launch = [&](void* out, int64_t start, int64_t count, void* inst, int blocks) {
            RadTensor ts[] = {
                T((uint16_t*)input + guard + start * sp, RAD_BF16, { count, W }),
                T((uint16_t*)out + guard + start * dp, RAD_BF16, { count, 4 * W }),
            };
            ts[0].stride[0] = sp; ts[1].stride[0] = dp;
            const RadParam ps[] = { PI("world_size", 4), PI("numel", count * W), PS("dtype", "bf16"),
                                    PI("row", W), PI("nblocks", blocks), PI("nthreads", 256) };
            RadArgs a{};
            a.t = ts; a.n_t = 2; a.p = ps; a.n_p = 6; a.instance = inst;
            a.rank = t->rank; a.world_size = 4;
            return t->row->launch(&a, stream);
        };
        auto slices = [&](int64_t n, int blocks) {
            int rc = RAD_OK;
            for (int64_t at = 0; at < n && rc >= 0; at += t->cap)
                rc = launch(part, r0 + at, std::min<int64_t>(t->cap, n - at), t->sliced, blocks);
            return rc;
        };
        auto prepare = [&](int64_t n, unsigned epoch) {
            std::fill(host.begin(), host.end(), 0x5AA5);
            for (int64_t r = r0; r < r0 + n; ++r)
                for (int64_t c = 0; c < W; ++c)
                    host[(size_t)(guard + r * sp + c)] = ag_rows_bits(t->rank, r, c, epoch);
            if (hipMemcpyAsync(input, host.data(), ni * 2, hipMemcpyHostToDevice, stream) != hipSuccess ||
                hipMemsetAsync(full, 0xA5, no * 2, stream) != hipSuccess ||
                hipMemsetAsync(part, 0xA5, no * 2, stream) != hipSuccess ||
                st_stream_sync(stream) != hipSuccess) t->rc = RAD_E_DEVICE;
        };
        auto check = [&](int64_t n, unsigned epoch) {
            if (st_stream_sync(stream) != hipSuccess) { t->rc = RAD_E_DEVICE; return; }
            st_memcpy(gold.data(), full, no * 2, hipMemcpyDeviceToHost);
            st_memcpy(got.data(), part, no * 2, hipMemcpyDeviceToHost);
            for (size_t i = 0; i < no; ++i) {
                uint16_t want = 0xA5A5;
                if (i >= (size_t)guard && i < (size_t)(guard + rows * dp)) {
                    const int64_t r = ((int64_t)i - guard) / dp, c = ((int64_t)i - guard) % dp;
                    if (r >= r0 && r < r0 + n && c < 4 * W)
                        want = ag_rows_bits((int)(c / W), r, c % W, epoch);
                }
                t->bad += got[i] != gold[i];
                t->bad += got[i] != want;  // checks full placement, row padding, prefix/tail and guards
            }
            // Reuse the now-checked larger gold buffer for the unchanged source, guards included.
            st_memcpy(gold.data(), input, ni * 2, hipMemcpyDeviceToHost);
            t->bad += std::memcmp(gold.data(), host.data(), ni * 2) != 0;
        };
        // 20/21 source rows straddle the four-rank full-fence threshold (131072 elements).
        // cap+20/cap+21 also exercise these as the LAST slice following a large message.
        const int sizes[] = { 4096, 1, 20, 21, 63, 64, 65, 1023, 1024, 1025,
                              t->cap + 20, t->cap + 21, 2048, 4096 };
        for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]) && t->rc == RAD_OK; ++k) {
            const int64_t n = sizes[k];
            const unsigned epoch = (unsigned)k + 1;
            prepare(n, epoch);
            const int blocks = 2 << (k % 4);  // blocks disappear and return; ranges change with n
            int rc = launch(full, r0, n, t->whole, blocks);
            if (rc >= 0) rc = slices(n, blocks);
            if (rc < 0) t->rc = rc;
            check(n, epoch);
        }
        // Capture the COMPLETE sliced sequence once, then change input bytes between replays.
        // Reset idle instances near wrap so graph replay crosses wrap and both slot parities too.
        t->idle_rc = t->rc;
        t->barrier->wait();  // all ranks' streams must be idle before ANY rank resets its flags
        bool capture_ready = true;
        for (const auto& r : *t->all) capture_ready = capture_ready && r.idle_rc == RAD_OK;
        if (capture_ready) {
            t->rc = t->seed(t->whole, t->handshake, t->parity);
            if (t->rc == RAD_OK) t->rc = t->seed(t->sliced, t->handshake, t->parity);
        }
        t->seed_rc = t->rc;
        t->barrier->wait();
        for (const auto& r : *t->all) capture_ready = capture_ready && r.seed_rc == RAD_OK;
        if (capture_ready) {
            const hipError_t begin = hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);
            int rc = begin == hipSuccess ? slices(C, 8) : RAD_E_DEVICE;
            const hipError_t end = begin == hipSuccess ? hipStreamEndCapture(stream, &graph) : begin;
            if (rc < 0 || end != hipSuccess || hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != hipSuccess)
                t->rc = RAD_E_DEVICE;
        }
        t->capture_rc = t->rc;
        t->barrier->wait();  // never replay on one rank if a peer failed to capture/instantiate
        for (const auto& r : *t->all) capture_ready = capture_ready && r.capture_rc == RAD_OK;
        if (capture_ready) for (unsigned replay = 0; replay < 3; ++replay) {
            const unsigned epoch = 1000 + replay;
            prepare(C, epoch);
            const int rc = launch(full, r0, C, t->whole, 8);
            if (rc < 0 || hipGraphLaunch(exec, stream) != hipSuccess) t->rc = RAD_E_DEVICE;
            check(C, epoch);
        }
    }
    if (exec) hipGraphExecDestroy(exec);
    if (graph) hipGraphDestroy(graph);
    for (void* p : { input, full, part }) if (p) st_free(p);
    if (stream) hipStreamDestroy(stream);
}
static void ds41_ag_rows_exact(const Plugin& pl, const std::vector<int>& devs) {
    const char* name = "ds41_ag_rows_exact";
    if (devs.size() < 4 || !ar_peer_path_ok(devs, 4)) { skip(name, "needs four P2P cards"); return; }
    const auto* row = find(pl, "ar_gather_4rank");
    const auto seed = (ArSeedSeq)dlsym(pl.h, "r4d_ar_test_seed_seq");
    if (!row || !seed) { ok(name, false, "missing gather row or idle sequence seed hook"); return; }
    for (int cap : { 1024, 64 }) for (bool pitched : { false, true }) {
        std::vector<AgRowsRank> ranks(4);
        StBarrier barrier(4);
        std::vector<std::thread> threads;
        for (int rank = 0; rank < 4; ++rank) {
            auto& r = ranks[(size_t)rank];
            r.row = row; r.seed = seed; r.barrier = &barrier; r.all = &ranks;
            r.rank = rank; r.dev = devs[(size_t)rank]; r.cap = cap; r.pitched = pitched;
            r.handshake = pitched ? 0xFFFFFFFEu : 0x7FFFFFFEu;
            r.parity = (cap == 64) ^ pitched;
            threads.emplace_back(ag_rows_worker, &r);
        }
        for (auto& thread : threads) thread.join();
        size_t bad = 0;
        bool up = true;
        for (const auto& r : ranks) { bad += r.bad; up = up && r.rc == RAD_OK; }
        ok((std::string(name) + " cap " + std::to_string(cap) + ", " + (pitched ? "pitched" : "tight")).c_str(),
           up && bad == 0, std::to_string(bad) + " placement/byte/guard/source differences; fence tails, wrap, graph replay");
        for (const auto& r : ranks) {
            if (r.whole) row->fini(r.whole);
            if (r.sliced) row->fini(r.sliced);
        }
    }
}
