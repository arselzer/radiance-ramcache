#!/usr/bin/env bash
# aux_yield.sh <control path> [timeout s, 60] -- <command> ...
#
# Runs <command> only once the radiance server whose RADIANCE_EXPERIMENTAL_DS41_AUX_CONTROL is <control path> has given
# the aux worker's card back, and keeps it given back while the command runs (work/aux-dyn/SPEC.md A.7). For jobs that
# need the extra card's memory (a training run, an eval): without it, a job that allocates more than the card's free
# VRAM before the server's watch releases the arena can fail its first allocation.
#
#   1. A holder: <path>.d/job-<pid>-<ms>, made under a dot name, locked shared (flock), then renamed into place, so the
#      server never sees an unlocked holder of a live job. The server's mode is off while any holder's lock is held
#      (the kernel drops it with this script, a crash included; it works across pid namespaces on a shared path).
#   2. It waits until the server's state file <path>.state says, in a line written after the holder appeared,
#        released ...        the arena freed (the server keeps serving on the CPU path)
#        dead freed ...      the worker is off for the server's life and its arena was freed
#        stopped ...         the server exited
#      or until no server holds <path>.lock (none started, or it exited: the driver frees a dead process's memory).
#   3. It never runs the command when the state says `dead held <MiB>` (the arena stays allocated until the server
#      restarts: exit 3), or when the timeout passes first (exit 4); either prints the state line.
#   4. The command runs with the holder held (its own descriptors do not carry the lock); the holder goes when it ends.
#      Exit: the command's status.
#
# Several holders overlap safely: the server reacquires (after its reacquire delay) only once every holder is gone.
set -u
usage() { echo "usage: aux_yield.sh <control path> [timeout s, default 60] -- <command> ..." >&2; exit 2; }
(( $# >= 3 )) || usage
CTL=$1; shift
TIMEOUT=60
if [[ $1 != -- ]]; then TIMEOUT=$1; shift; fi
[[ ${1:-} == -- ]] || usage
shift
(( $# > 0 )) || usage
[[ $TIMEOUT =~ ^[0-9]+$ ]] || usage
D=$CTL.d STATE=$CTL.state LOCK=$CTL.lock
mkdir -p "$D" || exit 2
# Unix ms from bash's own clock (some date(1) builds ignore %3N's width and print nanoseconds).
now_ms() { local t=${EPOCHREALTIME//[!0-9]/}; echo $(( 10#$t / 1000 )); }
T0=$(now_ms)
label=job-$$-$T0
tmp=$D/.tmp-$label
exec 8>"$tmp" || exit 2
flock -s 8 || exit 2
mv -f "$tmp" "$D/$label" || exit 2
trap 'rm -f "$D/$label"' EXIT

# 0 when a server holds its lock (it keeps an exclusive flock on <path>.lock for its life).
server_alive() {
  [[ -e $LOCK ]] || return 1
  ! flock -n -s "$LOCK" true 2>/dev/null
}
# 0 when the state line's "pid P S" names a live process with that start time (for a server without the lock file).
pid_alive() {
  local pid st
  [[ $1 =~ \ pid\ ([0-9]+)\ ([0-9]+) ]] || return 1
  pid=${BASH_REMATCH[1]} st=${BASH_REMATCH[2]}
  [[ -r /proc/$pid/stat ]] || return 1
  [[ $(sed -E 's/^.*\) //' "/proc/$pid/stat" | awk '{ print $20 }') == "$st" ]]
}

deadline=$(( T0 + TIMEOUT * 1000 ))
line=""
while :; do
  line=$(head -n1 "$STATE" 2>/dev/null)
  if [[ -e $LOCK ]]; then
    server_alive || { echo "aux_yield: no server holds $LOCK; running" >&2; break; }
  elif ! pid_alive "$line"; then
    echo "aux_yield: no server (${line:-no state file}); running" >&2
    break
  fi
  read -r w1 w2 w3 _ <<< "$line"
  ms=0
  case ${w1:-} in
    released|stopped) ms=${w2:-0} ;;
    dead)
      if [[ ${w2:-} == held ]]; then
        echo "aux_yield: the server holds the card until it restarts; not running: $line" >&2
        exit 3
      fi
      [[ ${w2:-} == freed ]] && ms=${w3:-0} ;;
  esac
  [[ $ms =~ ^[0-9]+$ ]] || ms=0
  if (( ms >= T0 )); then
    echo "aux_yield: the card is given back ($w1); running" >&2
    break
  fi
  if (( $(now_ms) >= deadline )); then
    echo "aux_yield: the card was not given back within $TIMEOUT s; not running: ${line:-no state file}" >&2
    exit 4
  fi
  sleep 0.1
done
"$@" 8>&-
