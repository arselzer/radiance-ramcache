/* engine_hostmem.h -- the memory cgroup limits bring-up checks before it locks a table into host
 * memory (engine_bringup.cpp, lockable_room). */
#pragma once
#include <cstdint>
#include <functional>
#include <string>

namespace rad {

/* A cgroup's limit less the usage reclaim cannot give back: memory.current less file pages (v2)
 * or usage_in_bytes less cache (v1), since page cache yields to a locked allocation and anonymous
 * memory does not. "max" and v1's near-INT64_MAX sentinel mean no limit. True, with `room` set,
 * when `dir` has a limit. */
bool cgroup_room(const std::string& dir, bool v2, int64_t* room);

/* Every level of the memory cgroup at `path` (as /proc/self/cgroup names it), from that level up
 * to the hierarchy's root mounted at `root`, leaf first: `visit(dir, room)` for each level with a
 * limit, until it returns false. Each level is a limit of its own. */
void cgroup_levels(const std::string& root, std::string path, bool v2,
                   const std::function<bool(const std::string& dir, int64_t room)>& visit);

/* Whether `need` more bytes fit in host memory after `still` more this process will allocate
 * first and `margin` (--host-headroom-mib) stays free, or why not. Checks MemAvailable; every
 * memory cgroup from the process's own up to the visible root (Docker's --memory, enforced by the
 * OOM killer); and, for an mlock()ed copy (`mlocks`), RLIMIT_MEMLOCK unless the process has
 * CAP_IPC_LOCK. Pinned memory a card reads is a driver userptr mapping, which RLIMIT_MEMLOCK does
 * not count. An unreadable limit is ignored, never guessed. */
bool lockable_room(int64_t need, int64_t still, int64_t margin, bool mlocks, std::string* why);

/* The margin a startup check keeps free (Engine::host_headroom_bytes): --host-headroom-mib plus
 * the part of the expert RAM cache's budget (--expert-ram-cache-total-mib) its slabs have not
 * backed yet, which a cache growing on demand takes later. `backed_bytes` is what the slabs
 * already hold: that memory is already gone from MemAvailable, so reserving it again counts it
 * twice. With every slab pinned at load (RADIANCE_EXPERIMENTAL_NATIVE_CACHE_PREGROW) the whole
 * budget is counted twice, and a 96 GiB cache on a 123 GiB host leaves the checkpoint store no
 * room at all. `backed_bytes` 0 is the whole budget, as the margin always was. INT64_MAX for a
 * negative figure or a sum past int64. */
int64_t host_headroom_total(int64_t headroom_mib, int64_t cache_mib, int64_t backed_bytes);

/* RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET (engine_rambudget.cpp), defined in engine_bringup.cpp
 * beside the checks they reuse. host_room_now: MemAvailable, or less where a memory cgroup leaves
 * less, with the binding limit in `why`; -1 when MemAvailable is unreadable. host_pin_room_now:
 * what the amdgpu driver's pin cap leaves this process (its cap less everything this process has
 * pinned or mapped through rad_dev_alloc, a ledger that never counts down, so it errs low);
 * INT64_MAX when the cap is lifted or unknown, or on a host without cards. */
int64_t host_room_now(std::string* why);
int64_t host_pin_room_now();

}  /* namespace rad */
