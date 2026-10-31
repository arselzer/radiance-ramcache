/* Test only; included after the independent AR/AG checks in r4d_selftest.cpp. Equal bounds
 * must acquire one allocation family, regardless of the init row or gather placement mode. */
struct SharedArAgFrame {
    int kind = 0;                         // 0 AR one, 1 AR two, 2 AG concat, 3 AG rows/pitched
    int64_t width = 0, sp = 0, dp = 0, capacity = 0, count = 0;
    size_t ni = 0, no = 0;
    bool in_place = false;
    void *in_shared = nullptr, *in_control = nullptr, *out_shared = nullptr, *out_control = nullptr;
    std::vector<uint16_t> host, gold, got;
};
struct SharedArAgRank {
    const RadKernelInfo *one = nullptr, *two = nullptr, *ag = nullptr;
    ArSeedSeq seed = nullptr;
    StBarrier* barrier = nullptr;
    std::vector<SharedArAgRank>* all = nullptr;
    int rank = 0, dev = 0, seed_case = 0;
    bool reverse = false;
    void* aliases[4] = {};                // each owns exactly one successful init reference
    void* shared = nullptr;               // NON-owning: never fini this separately
    void *control_ar = nullptr, *control_ag = nullptr;
    int rc = RAD_OK, setup_rc = RAD_E_STATE, seed_rc = RAD_E_STATE;
    int idle_rc = RAD_E_STATE, reacquire_rc = RAD_E_STATE, capture_rc = RAD_E_STATE;
    size_t bad = 0, fold_bad = 0;
};
static void ds41_ar_ag_shared_worker(SharedArAgRank* t) {
    constexpr int64_t D = 5120, W = 6400, AR = 1280, AG = 1024, N = AR * D;
    static_assert(N == AG * W, "the coordinated caps must have the same largest bound");
    constexpr int64_t first = 13, guard = 128;
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    SharedArAgFrame frames[4];
    t->rc = st_set_device(t->dev) == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    if (t->rc == RAD_OK && hipStreamCreate(&stream) != hipSuccess) t->rc = RAD_E_DEVICE;
    auto init_alias = [&](int kind, int64_t bound, void** out) {
        // exact/min_bytes/row differ deliberately; none may partition the allocation key.
        const RadParam ar[] = { PI("world_size", 4), PI("numel", bound), PS("dtype", "bf16"),
                                PI("exact", 1), PI("min_bytes", kind == 0 ? 0 : 131072) };
        const RadParam ag[] = { PI("world_size", 4), PI("numel", bound), PS("dtype", "bf16"),
                                PI("row", kind == 2 ? 0 : W) };
        const auto* row = kind == 0 ? t->one : kind == 1 ? t->two : t->ag;
        return row->init(kind < 2 ? ar : ag, kind < 2 ? 5 : 4, t->rank, 4, out);
    };
    for (int i = 0; i < 4 && t->rc == RAD_OK; ++i) {
        const int kind = t->reverse ? 3 - i : i;
        t->rc = init_alias(kind, N, &t->aliases[kind]);
    }
    t->shared = t->aliases[0];
    if (t->rc == RAD_OK) {
        for (void* alias : t->aliases) if (!alias || alias != t->shared) t->rc = RAD_E_STATE;
    }
    if (t->rc == RAD_OK) t->rc = init_alias(1, N + 64, &t->control_ar);
    if (t->rc == RAD_OK) t->rc = init_alias(3, N + 128, &t->control_ag);
    if (t->rc == RAD_OK && (t->control_ar == t->shared || t->control_ag == t->shared ||
                            t->control_ar == t->control_ag)) t->rc = RAD_E_STATE;
    for (int kind = 0; kind < 4 && t->rc == RAD_OK; ++kind) {
        auto& f = frames[kind];
        f.kind = kind; f.width = kind < 2 ? D : W; f.capacity = kind < 2 ? AR : AG;
        f.sp = f.width + (kind == 3 ? 64 : 0);
        f.dp = (kind < 2 ? D : 4 * W) + (kind == 3 ? 64 : 0);
        f.ni = (size_t)(2 * guard + (f.capacity + 2 * first) * f.sp);
        f.no = (size_t)(2 * guard + (f.capacity + 2 * first) * f.dp);
        if (hipMalloc(&f.in_shared, f.ni * 2) != hipSuccess || hipMalloc(&f.in_control, f.ni * 2) != hipSuccess ||
            hipMalloc(&f.out_shared, f.no * 2) != hipSuccess || hipMalloc(&f.out_control, f.no * 2) != hipSuccess)
            t->rc = RAD_E_NOMEM;
        else { f.host.resize(f.ni); f.gold.resize(f.no); f.got.resize(f.no); }
    }
    t->setup_rc = t->rc;
    t->barrier->wait();
    bool ready = true;
    for (const auto& r : *t->all) ready = ready && r.setup_rc == RAD_OK;
    if (ready) {
        // Release the first-acquired alias; all mixed launches below use the surviving aliases.
        const int released = t->reverse ? 3 : 0;
        (released < 2 ? t->one : t->ag)->fini(t->aliases[released]);
        t->aliases[released] = nullptr;
        const unsigned handshake = t->seed_case ? 0xFFFFFFFEu : 0x7FFFFFFEu;
        auto seed_idle = [&] {
            int rc = t->seed(t->shared, handshake, (unsigned)t->seed_case); // shared pointer ONCE
            if (rc == RAD_OK) rc = t->seed(t->control_ar, handshake, (unsigned)t->seed_case);
            if (rc == RAD_OK) rc = t->seed(t->control_ag, handshake, (unsigned)t->seed_case);
            return rc;
        };
        t->rc = seed_idle();
        t->seed_rc = t->rc;
        t->barrier->wait();
        bool seeded = true;
        for (const auto& r : *t->all) seeded = seeded && r.seed_rc == RAD_OK;
        const int ts_min = t->seed_case ? 32768 : 0;
        auto prepare = [&](SharedArAgFrame& f, int64_t count, bool in_place, unsigned epoch) {
            f.count = count; f.in_place = f.kind < 2 && in_place;
            std::fill(f.host.begin(), f.host.end(), 0x5AA5);
            for (int64_t r = first; r < first + count; ++r)
                for (int64_t c = 0; c < f.width; ++c)
                    f.host[(size_t)(guard + r * f.sp + c)] = f.kind < 2
                        ? ar_rows_value(t->rank, r * D + c + (int64_t)epoch * 257)
                        : ag_rows_bits(t->rank, r, c, epoch);
            if (hipMemcpyAsync(f.in_shared, f.host.data(), f.ni * 2, hipMemcpyHostToDevice, stream) != hipSuccess ||
                hipMemcpyAsync(f.in_control, f.host.data(), f.ni * 2, hipMemcpyHostToDevice, stream) != hipSuccess ||
                hipMemsetAsync(f.out_shared, 0xA5, f.no * 2, stream) != hipSuccess ||
                hipMemsetAsync(f.out_control, 0xA5, f.no * 2, stream) != hipSuccess ||
                st_stream_sync(stream) != hipSuccess) t->rc = RAD_E_DEVICE;
        };
        auto launch = [&](SharedArAgFrame& f, bool control, int blocks) {
            void* input = control ? f.in_control : f.in_shared;
            void* output = f.in_place ? input : control ? f.out_control : f.out_shared;
            RadTensor tensors[] = {
                T((uint16_t*)input + guard + first * f.sp, RAD_BF16, { f.count, f.width }),
                T((uint16_t*)output + guard + first * f.dp, RAD_BF16,
                  { f.count, f.kind < 2 ? D : 4 * W }),
            };
            tensors[0].stride[0] = f.sp; tensors[1].stride[0] = f.dp;
            // row=0 is deliberately a plain concatenating gather; do not reinterpret it as rows.
            if (f.kind == 2) {
                tensors[0] = T(tensors[0].data, RAD_BF16, { f.count * W });
                tensors[1] = T(tensors[1].data, RAD_BF16, { f.count * W * 4 });
            }
            const RadParam ps[] = {
                PI("world_size", 4), PI("numel", f.count * f.width), PS("dtype", "bf16"),
                PI("exact", 1), PI("row", f.kind == 3 ? W : 0),
                PI("nblocks", blocks), PI("nthreads", 256), PI("ts_min", ts_min),
            };
            RadArgs a{};
            a.t = tensors; a.n_t = f.in_place ? 1 : 2; a.p = ps; a.n_p = 8;
            a.instance = control ? (f.kind < 2 ? t->control_ar : t->control_ag) : t->shared;
            a.rank = t->rank; a.world_size = 4;
            const auto* row = f.kind == 0 ? t->one : f.kind == 1 ? t->two : t->ag;
            return row->launch(&a, stream);
        };
        auto check = [&](SharedArAgFrame& f, unsigned epoch) {
            if (st_stream_sync(stream) != hipSuccess) { t->rc = RAD_E_DEVICE; return; }
            void* a = f.in_place ? f.in_shared : f.out_shared;
            void* b = f.in_place ? f.in_control : f.out_control;
            st_memcpy(f.got.data(), a, f.no * 2, hipMemcpyDeviceToHost);
            st_memcpy(f.gold.data(), b, f.no * 2, hipMemcpyDeviceToHost);
            const int64_t begin = guard + first * f.dp;
            for (size_t i = 0; i < f.no; ++i) {
                t->bad += f.got[i] != f.gold[i];  // every bit, including all exceptional payloads
                const int64_t q = (int64_t)i - begin;
                const bool active = q >= 0 && q / f.dp < f.count && q % f.dp < (f.kind < 2 ? D : 4 * W);
                if (!active) {
                    const uint16_t want = f.in_place ? f.host[i] : 0xA5A5;
                    t->bad += f.got[i] != want;
                } else if (f.kind >= 2) {
                    int rank = 0; int64_t r = 0, c = 0;
                    if (f.kind == 2) { // [rank][all rows][columns], for THIS single message
                        rank = (int)(q / (f.count * W)); r = first + q % (f.count * W) / W; c = q % W;
                    } else { rank = (int)(q % f.dp / W); r = first + q / f.dp; c = q % f.dp % W; }
                    t->bad += f.got[i] != ag_rows_bits(rank, r, c, epoch);
                } else if (q % 127 == 0) {
                    const int64_t index = (first + q / D) * D + q % D + (int64_t)epoch * 257;
                    volatile float acc = 0.0f;
                    bool finite = true;
                    for (int rank = 0; rank < 4; ++rank) {
                        const uint16_t bits = ar_rows_value(rank, index);
                        finite = finite && (bits & 0x7F80) != 0x7F80;
                        acc = acc + h_bf16(bits);
                    }
                    if (finite) t->fold_bad += f.got[i] != h_to_bf16(acc);
                }
            }
            if (!f.in_place) { // both original inputs, including every source/padding guard, survive
                st_memcpy(f.gold.data(), f.in_shared, f.ni * 2, hipMemcpyDeviceToHost);
                t->bad += std::memcmp(f.gold.data(), f.host.data(), f.ni * 2) != 0;
                st_memcpy(f.gold.data(), f.in_control, f.ni * 2, hipMemcpyDeviceToHost);
                t->bad += std::memcmp(f.gold.data(), f.host.data(), f.ni * 2) != 0;
            }
        };
        if (seeded) {
            struct Step { int type; int rows; }; // 0 AR, 2 concat gather, 3 row/pitched gather
            const Step steps[] = {
                {0,1}, {2,1}, {0,6}, {3,20}, {0,7}, {2,20}, {0,25}, {3,21},
                {0,26}, {2,21}, {0,102}, {3,1023}, {0,103}, {2,1024},
                {0,1279}, {3,1025}, {0,1280}, {2,1025}, {0,1281}, {3,1044},
                {0,1305}, {2,1045}, {0,1306}, {3,1045}, {0,2048}, {2,2048},
                {0,4096}, {3,4096},
            };
            unsigned epoch = 0;
            for (const auto& step : steps) for (int at = 0; at < step.rows && t->rc == RAD_OK;) {
                const int64_t count = std::min<int64_t>(step.type == 0 ? AR : AG, step.rows - at);
                const int kind = step.type == 0 ? (count * D <= 524288 ? 0 : 1) : step.type;
                auto& f = frames[kind];
                ++epoch;
                prepare(f, count, epoch % 2, epoch);
                const int blocks = 2 << (epoch % 4);
                int rc = launch(f, false, blocks);
                if (rc >= 0) rc = launch(f, true, blocks);
                if (rc < 0) t->rc = rc;
                check(f, epoch);
                at += (int)count;
            }
        }
        t->idle_rc = t->rc;
        t->barrier->wait(); // no rank may reseed while any peer still reads the shared slots
        bool capture_ready = seeded;
        for (const auto& r : *t->all) capture_ready = capture_ready && r.idle_rc == RAD_OK;
        if (capture_ready) {
            t->rc = init_alias(released, N, &t->aliases[released]);
            if (t->rc == RAD_OK && t->aliases[released] != t->shared) t->rc = RAD_E_STATE;
            if (t->rc == RAD_OK) t->rc = seed_idle();
        }
        t->reacquire_rc = t->rc;
        t->barrier->wait();
        for (const auto& r : *t->all) capture_ready = capture_ready && r.reacquire_rc == RAD_OK;
        const int order[] = { 0, 2, 1, 3 };
        if (capture_ready) {
            for (int kind : order) prepare(frames[kind], kind == 0 ? 102 : kind == 1 ? AR : AG,
                                           kind == 0, 1000);
            const hipError_t begin = hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);
            int rc = begin == hipSuccess ? RAD_OK : RAD_E_DEVICE;
            if (rc == RAD_OK) for (int kind : order) {
                rc = launch(frames[kind], false, 2 << kind);
                if (rc >= 0) rc = launch(frames[kind], true, 2 << kind);
                if (rc < 0) break;
            }
            const hipError_t end = begin == hipSuccess ? hipStreamEndCapture(stream, &graph) : begin;
            if (rc < 0 || end != hipSuccess || hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != hipSuccess)
                t->rc = RAD_E_DEVICE;
        }
        t->capture_rc = t->rc;
        t->barrier->wait();
        for (const auto& r : *t->all) capture_ready = capture_ready && r.capture_rc == RAD_OK;
        if (capture_ready) for (unsigned replay = 0; replay < 3; ++replay) {
            for (int kind : order) prepare(frames[kind], kind == 0 ? 102 : kind == 1 ? AR : AG,
                                           kind == 0, 1000 + replay);
            if (hipGraphLaunch(exec, stream) != hipSuccess) t->rc = RAD_E_DEVICE;
            for (int kind : order) check(frames[kind], 1000 + replay);
        }
    }
    if (exec) hipGraphExecDestroy(exec);
    if (graph) hipGraphDestroy(graph);
    for (const auto& f : frames)
        for (void* p : { f.in_shared, f.in_control, f.out_shared, f.out_control }) if (p) st_free(p);
    if (stream) hipStreamDestroy(stream);
}
static void ds41_ar_ag_shared_exact(const Plugin& pl, const std::vector<int>& devs) {
    const char* name = "ds41_ar_ag_shared_exact";
    if (devs.size() < 4 || !ar_peer_path_ok(devs, 4)) { skip(name, "needs four P2P cards"); return; }
    const auto *one = find(pl, "ar_oneshot_4rank_exact"), *two = find(pl, "ar_twoshot_4rank_exact"),
               *ag = find(pl, "ar_gather_4rank");
    const auto seed = (ArSeedSeq)dlsym(pl.h, "r4d_ar_test_seed_seq");
    if (!one || !two || !ag || !seed) { ok(name, false, "missing exact rows or idle sequence seed hook"); return; }
    for (bool reverse : { false, true }) for (int seed_case : { 0, 1 }) {
        std::vector<SharedArAgRank> ranks(4);
        StBarrier barrier(4);
        std::vector<std::thread> threads;
        for (int rank = 0; rank < 4; ++rank) {
            auto& r = ranks[(size_t)rank];
            r.one = one; r.two = two; r.ag = ag; r.seed = seed; r.barrier = &barrier; r.all = &ranks;
            r.rank = rank; r.dev = devs[(size_t)rank]; r.reverse = reverse; r.seed_case = seed_case;
            threads.emplace_back(ds41_ar_ag_shared_worker, &r);
        }
        for (auto& thread : threads) thread.join();
        size_t bad = 0, fold_bad = 0;
        bool up = true;
        for (const auto& r : ranks) { bad += r.bad; fold_bad += r.fold_bad; up = up && r.rc == RAD_OK; }
        ok((std::string(name) + " order " + (reverse ? "ag-first" : "ar-first") + ", seed " +
            std::to_string(seed_case)).c_str(), up && !bad && !fold_bad,
           std::to_string(bad) + " byte/placement/guard/source differences, " + std::to_string(fold_bad) +
           " finite-fold samples; shared pointers, released live alias, thresholds/tails, wrap, mixed graph replay");
        for (const auto& r : ranks) {
            st_set_device(r.dev);
            for (void* alias : r.aliases) if (alias) one->fini(alias); // one fini per owning init
            if (r.control_ar) one->fini(r.control_ar);
            if (r.control_ag) ag->fini(r.control_ag);
        }
    }
}
