# Sourced by ab_window.sh. No checker runs here or outside that controller's owned stopped window.
# A separate explicit build-owner manifest extends the existing full-cohort manifest:
# ds41-window-kv-cohort-v1; source_commit40hex; packed_cohort64hex (full manifest SHA);
# codec_checker64hex; paths_checker64hex. Logs never infer source/config authority.
window_kv_cohort_key() { # homes, served library, fast0/1
  local home=$1 lib=$2 mode=$3 cohort key value extra actual source
  local -A fields=()
  cohort=$(kv_cohort_key "$home" "$lib" "$mode") || return 1
  [[ -f ${WINDOW_KV_COHORT_MANIFEST:-} && -x ${WINDOW_KV_CODEC_CHECK:-} && -x ${WINDOW_KV_PATHS_CHECK:-} ]] || return 1
  [[ $(head -1 "$WINDOW_KV_COHORT_MANIFEST") == ds41-window-kv-cohort-v1 ]] || return 1
  while read -r key value extra; do
    [[ -n $key && -n $value && -z ${extra:-} ]] || return 1
    case $key in
      source_commit) [[ $value =~ ^[0-9a-f]{40}$ ]] || return 1 ;;
      packed_cohort|codec_checker|paths_checker) [[ $value =~ ^[0-9a-f]{64}$ ]] || return 1 ;;
      *) return 1 ;;
    esac
    [[ -z ${fields[$key]+present} ]] || return 1
    fields[$key]=$value
  done < <(tail -n +2 "$WINDOW_KV_COHORT_MANIFEST")
  [[ ${#fields[@]} == 4 && ${fields[packed_cohort]:-} == "$cohort" ]] || return 1
  source=$(awk '$1=="source_commit"{print $2}' "$KV_COHORT_MANIFEST")
  [[ ${fields[source_commit]:-} == "$source" ]] || return 1
  actual=$(sha256sum "$WINDOW_KV_CODEC_CHECK" | cut -d' ' -f1)
  [[ $actual == "${fields[codec_checker]:-}" ]] || return 1
  actual=$(sha256sum "$WINDOW_KV_PATHS_CHECK" | cut -d' ' -f1)
  [[ $actual == "${fields[paths_checker]:-}" ]] || return 1
  sha256sum "$WINDOW_KV_COHORT_MANIFEST" | cut -d' ' -f1
}

window_kv_scope() {
  [[ ${KV_WINDOW_ACTIVE:-} == 1 && ${KV_PROOFS_OPEN:-} == 1 && -n ${KV_WINDOW_TOKEN:-} ]] || return 1
  [[ $("${DOCKER[@]}" inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null) == exited ]] || return 1
  [[ $(cat /home/as/ds41-claude-build/work/qfn_stopped_by 2>/dev/null) == $$ ]] || return 1
  # Occupied locks plus this live controller's QFN ownership; never acquire them out of order.
  [[ -f "$R/research-compiler.lock" && -f "$R/research-heavy-runtime.lock" ]] || return 1
  ! flock -n "$R/research-compiler.lock" true || return 1
  ! flock -n "$R/research-heavy-runtime.lock" true || return 1
}

window_kv_receipt_ok() { # file, full library SHA, extended cohort SHA, fast mode
  local f=$1 sum=$2 cohort=$3 mode=$4 card profile body input_profile
  [[ -f $f && -n ${KV_WINDOW_TOKEN:-} && ( $mode == 0 || $mode == 1 ) ]] || return 1
  [[ $(head -1 "$f") == "window-kv receipt v1 token=$KV_WINDOW_TOKEN lib=$sum cohort=$cohort fast=$mode" ]] || return 1
  [[ $(tail -1 "$f") == "window-kv finished token=$KV_WINDOW_TOKEN lib=$sum cohort=$cohort fast=$mode" ]] || return 1
  [[ $(grep -Fxc 'window-kv exit 0' "$f") == 1 ]] || return 1
  ! grep -Eq '(^|[[:space:]])(FAIL|SKIP|skip)([[:space:]:]|$)' "$f" || return 1
  [[ $(grep -c '^window-kv begin ' "$f") == 12 && $(grep -c '^window-kv end ' "$f") == 12 ]] || return 1
  for card in 0 1 2 3; do for profile in unpinned exact lcp; do
    [[ $(grep -Fxc "window-kv begin card=$card profile=$profile fast=$mode" "$f") == 1 &&
       $(grep -Fxc "window-kv end card=$card profile=$profile fast=$mode exit=0" "$f") == 1 ]] || return 1
    body=$(sed -n "/^window-kv begin card=$card profile=$profile fast=$mode$/,/^window-kv end card=$card profile=$profile fast=$mode exit=0$/p" "$f")
    [[ $(grep -Fxc 'PASS ds41_window_codec_exact' <<< "$body") == 2 ]] || return 1
    [[ $(grep -Ec '^PASS window-fp8-domain groups=1146 raw=0 bad=0 pair=[0-9]+$' <<< "$body") == 2 ]] || return 1
    [[ $(grep -Ec '^PASS window-fp8-original-producer profile=[0-3] groups=65536 raw=0 bad=0 scale_bad=0 byte_guard_bad=0 pair=[0-9]+$' <<< "$body") == 8 ]] || return 1
    for input_profile in 0 1 2 3; do
      [[ $(grep -Ec "^PASS window-fp8-original-producer profile=$input_profile groups=65536 raw=0 bad=0 scale_bad=0 byte_guard_bad=0 pair=[0-9]+$" <<< "$body") == 2 ]] || return 1
    done
    [[ $(grep -Fxc 'PASS window packed producers cases=84 raw=0 byte_errors=0 forms=7 max_M=4096' <<< "$body") == 1 ]] || return 1
    [[ $(grep -Fxc 'PASS window packed readers cases=1152 byte_errors=0 forms=32767' <<< "$body") == 1 ]] || return 1
    [[ $(grep -Fxc 'PASS ds41_window_packed_paths_exact producers=84 readers=1152 max_M=4096 raw=0 byte_errors=0' <<< "$body") == 1 ]] || return 1
  done; done
}

window_kv_check() { # homes, served library, fast mode; BEFORE all model arms only
  local home=$1 lib=$2 mode=$3 cohort sum key f card profile rc failed=0 after
  window_kv_scope || { log 'FP8 WINDOW checker refused: no owned two-lock stopped-QFN window'; return 1; }
  cohort=$(window_kv_cohort_key "$home" "$lib" "$mode") || { log 'FP8 WINDOW checker refused: explicit full cohort/checker hashes differ'; return 1; }
  sum=$(sha256sum "$lib" | cut -d' ' -f1); key=$sum:$cohort:$mode
  if [[ -n ${WINDOW_KV_PASSED[$key]+present} ]]; then
    [[ ${WINDOW_KV_PASSED[$key]} == 1 ]] && window_kv_receipt_ok "${WINDOW_KV_RECEIPT[$key]:-}" "$sum" "$cohort" "$mode"; return
  fi
  f=$OUT/window-kv-$sum-$cohort-fast$mode.log
  WINDOW_KV_PASSED[$key]=0 WINDOW_KV_RECEIPT[$key]=$f
  printf 'window-kv receipt v1 token=%s lib=%s cohort=%s fast=%s\n' "$KV_WINDOW_TOKEN" "$sum" "$cohort" "$mode" >"$f"
  for card in 0 1 2 3; do for profile in unpinned exact lcp; do
    window_kv_scope || { failed=1; break 2; }
    echo "window-kv begin card=$card profile=$profile fast=$mode" >>"$f"
    ( unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HSA_VISIBLE_DEVICES R4D_SELFTEST_ONE_CARD RADIANCE_TUNE
      while read -r knob; do unset "$knob"; done < <(compgen -v RADIANCE_EXPERIMENTAL_DS41_)
      export HIP_VISIBLE_DEVICES=$card RADIANCE_EXPERIMENTAL_DS41_KV_PACKED=1 RADIANCE_EXPERIMENTAL_DS41_WINDOW_PACKED=1
      export RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_FAST=$mode RADIANCE_EXPERIMENTAL_DS41_P1=0
      case $profile in
        unpinned) export RADIANCE_EXPERIMENTAL_DS41_N128=0 ;;
        exact) export RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1 RADIANCE_EXPERIMENTAL_DS41_N128=0 ;;
        lcp) export RADIANCE_EXPERIMENTAL_DS41_LCP=2048 RADIANCE_EXPERIMENTAL_DS41_N128=1 ;;
      esac
      timeout 1800 "$WINDOW_KV_CODEC_CHECK" "$lib" && timeout 1800 "$WINDOW_KV_PATHS_CHECK" "$lib"
    ) >>"$f" 2>&1
    rc=$?; (( rc == 0 )) || failed=1
    echo "window-kv end card=$card profile=$profile fast=$mode exit=$rc" >>"$f"
  done; done
  after=$(window_kv_cohort_key "$home" "$lib" "$mode") || failed=1
  [[ $after == "$cohort" && $(sha256sum "$lib" | cut -d' ' -f1) == "$sum" ]] || failed=1
  echo "window-kv exit $failed" >>"$f"
  echo "window-kv finished token=$KV_WINDOW_TOKEN lib=$sum cohort=$cohort fast=$mode" >>"$f"
  if (( failed == 0 )) && window_kv_receipt_ok "$f" "$sum" "$cohort" "$mode"; then WINDOW_KV_PASSED[$key]=1; fi
  log "FP8 WINDOW oracle fast$mode passed=${WINDOW_KV_PASSED[$key]} fullSHA=$sum cohort=$cohort receipt=$f"
  [[ ${WINDOW_KV_PASSED[$key]} == 1 ]]
}

window_kv_gate() { # homes, served library, requested fast mode; no GPU invocation here
  local home=$1 lib=$2 mode=$3 cohort sum key m
  cohort=$(window_kv_cohort_key "$home" "$lib" "$mode") || return 1
  sum=$(sha256sum "$lib" | cut -d' ' -f1)
  for m in 0 $([[ $mode == 1 ]] && echo 1); do
    key=$sum:$cohort:$m
    [[ ${WINDOW_KV_PASSED[$key]:-} == 1 ]] && window_kv_receipt_ok "${WINDOW_KV_RECEIPT[$key]:-}" "$sum" "$cohort" "$m" || return 1
  done
}
