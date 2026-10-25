#!/usr/bin/env python3
"""Parent-owned single GLM maintenance test. No execution at import/configuration.

prepare never changes a service. run-window owns only its immutable captured
radiance-flashnext ID and its newly spawned child process groups. No compose,
image pull/recreate, driver reset, deployment or configuration operation exists.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = 'radiance-flashnext'
HEALTH = 'http://127.0.0.1:8078/health'
NATIVE_BYTES = 161715585024
RECORDS = 48384
BUNDLE_ID = 7279810219167007678
MAX_JSON = 8 << 20
MAX_LOG = 64 << 20
STOP_SECONDS = 30
STOP_CLIENT_SECONDS = 90
RESTORE_SECONDS = 900
HEARTBEAT_SECONDS = 15
RESOURCE_WAIT_SECONDS = 60
MIN_AVAILABLE_BYTES = 32 << 30
MAX_IDLE_VRAM_BYTES = 64 << 20
EXPECTED_VRAM_BYTES = 17095983104
EXPECTED_AMD_BDFS = frozenset(('0000:c3:00.0', '0000:83:00.0', '0000:03:00.0', '0000:06:00.0'))
_INTERRUPTED = False
_RESTORING = False
_CLEANING = False
INSPECT_FORMAT = ('{"Id":{{json .Id}},"Image":{{json .Image}},"Name":{{json .Name}},'
                  '"Running":{{json .State.Running}},"Status":{{json .State.Status}},'
                  '"Paused":{{json .State.Paused}},"Restarting":{{json .State.Restarting}},'
                  '"Dead":{{json .State.Dead}},"ExitCode":{{json .State.ExitCode}},'
                  '"StartedAt":{{json .State.StartedAt}},"FinishedAt":{{json .State.FinishedAt}}}')


class Interrupted(RuntimeError):
    pass


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(value, why):
    if not value:
        raise RuntimeError(why)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and 0 < path.stat().st_size <= MAX_JSON, 'bounded JSON: ' + str(path))
    return json.loads(path.read_text())


def inside_root(path):
    path = Path(path).resolve(strict=True)
    require(ROOT in path.parents, 'research-owned existing path required: ' + str(path))
    return path


def verify_source():
    receipt = Path(__file__).parent / 'SOURCE_RECEIPT.json'
    data = read_json(receipt)
    require(data.get('schema') == 'glm53-manual-maintenance-source-v1' and
            data.get('runtime_executed_by_author') is False, 'frozen source-only runner receipt')
    require(set(data['source_files']) == {'runner.py', 'README.txt'}, 'exact runner source closure')
    for relative, expected in data['source_files'].items():
        require(sha(receipt.parent / relative) == expected, 'runner source drift: ' + relative)
    for relative, expected in data['inputs'].items():
        require(sha(inside_root(ROOT / relative)) == expected, 'runner reviewed input drift: ' + relative)
    return receipt


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def signal_handler(number, _frame):
    global _INTERRUPTED
    was_interrupted = _INTERRUPTED
    _INTERRUPTED = True
    if not was_interrupted and not _RESTORING and not _CLEANING:
        raise Interrupted('caught signal ' + str(number))


def install_signals():
    # Keep owned children waitable even if a launcher inherited SIGCHLD ignore.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, signal_handler)


def check_cancelled():
    if _INTERRUPTED and not _RESTORING:
        raise Interrupted('maintenance interrupted; no new test child permitted')


def safe_environment(gpus=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith('RADIANCE_')}
    visible = '0,1,2,3' if gpus else '99'
    env.update(ROCR_VISIBLE_DEVICES=visible,
               HIP_VISIBLE_DEVICES='0,1,2,3' if gpus else '-1',
               CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1',
               OMP_WAIT_POLICY='passive')
    return env


def child_exited_unreaped(proc):
    # WNOWAIT deliberately retains our direct child's PID reservation. This
    # prevents process-group ID reuse before descendant cleanup has completed.
    if proc.returncode is not None:
        return True
    info = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    return info is not None


def stop_owned(proc):
    global _CLEANING
    prior = _CLEANING
    _CLEANING = True  # a signal must not interrupt owned-child retirement
    try:
        if proc.returncode is not None:
            return
        # Confirm it is STILL our waitable child before targeting the group.
        # WNOWAIT keeps its PID reserved until descendant cleanup is finished.
        child_exited_unreaped(proc)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 10
        while not child_exited_unreaped(proc) and time.monotonic() < deadline:
            time.sleep(0.1)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=20)
    finally:
        _CLEANING = prior


def io_heartbeat(pid):
    out = {}
    try:
        for line in Path('/proc', str(pid), 'status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmPeak:')):
                key, raw = line.split(':', 1)
                out[key + '_kB'] = int(raw.split()[0])
        for line in Path('/proc', str(pid), 'io').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('read_bytes', 'write_bytes', 'rchar', 'wchar'):
                out[key] = int(value)
    except (OSError, ValueError):
        pass
    return out


def run_logged(command, output, timeout, env, receipt, save, guard=None):
    check_cancelled()
    entry = dict(command=[str(x) for x in command], log=str(output), started=now())
    receipt.setdefault('commands', []).append(entry)
    save()
    proc = None
    with Path(output).open('x') as handle:
        try:
            proc = subprocess.Popen(entry['command'], cwd=ROOT, env=env,
                                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            entry['owned_pid'] = proc.pid
            save()
            start = time.monotonic()
            heartbeat = start
            guarded = start
            while not child_exited_unreaped(proc):
                check_cancelled()
                elapsed = time.monotonic() - start
                if elapsed > timeout:
                    raise RuntimeError('owned child deadline: ' + str(output))
                if Path(output).stat().st_size > MAX_LOG:
                    raise RuntimeError('owned child log bound exceeded')
                if guard and time.monotonic() - guarded >= 5:
                    guard()
                    guarded = time.monotonic()
                if time.monotonic() >= heartbeat:
                    entry.update(heartbeat_utc=now(), elapsed_live=elapsed,
                                 os_counters=io_heartbeat(proc.pid))
                    print(json.dumps({'heartbeat': entry['heartbeat_utc'],
                                      'child_pid': proc.pid, 'elapsed_seconds': round(elapsed, 1),
                                      'log': str(output), **entry['os_counters']}), flush=True)
                    save()
                    heartbeat = time.monotonic() + HEARTBEAT_SECONDS
                time.sleep(0.2)
            stop_owned(proc)  # also retires any descendants before leader reap
            entry['returncode'] = proc.returncode
            entry['elapsed_seconds'] = time.monotonic() - start
            return proc.returncode
        finally:
            if proc is not None:
                stop_owned(proc)
                entry.setdefault('returncode', proc.returncode)
            entry['finished'] = now()
            save()


@contextlib.contextmanager
def lock_file(path, timeout=180):
    check_cancelled()
    with Path(path).open('a') as handle:
        deadline = time.monotonic() + timeout
        while True:
            check_cancelled()
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('research lock deadline: ' + str(path))
                time.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_build(path, backend, revision=None, mode="release"):
    path = inside_root(path)
    data = read_json(path)
    require(data.get('passed') is True and data.get('backend') == backend and data.get('mode') == mode,
            'successful matching ' + mode + ' ' + backend + ' build receipt required')
    rev = data.get('revision', '')
    require(re.fullmatch(r'v[1-9][0-9]*', rev) and (revision is None or rev == revision), 'same explicit Core revision')
    source = ROOT / ('radiance-glm-main-verified-source-' + rev + '-src')
    explicit_fixture = 'fixture_directory' in data or 'fixture_revision' in data
    if explicit_fixture:
        fixture_rev = data.get('fixture_revision', '')
        require(isinstance(fixture_rev, str) and re.fullmatch(r'v[1-9][0-9]*', fixture_rev),
                'explicit fixture revision required')
        require((rev, fixture_rev) == ('v6', 'v8') or (int(rev[1:]) <= 5 and fixture_rev == rev),
                'only reviewed Core/fixture pair v6/v8 or matching legacy <=v5')
        expected_fixture = ROOT / ('glm53-main-verified-source-engine-' + fixture_rev + '-prototype')
        supplied_fixture = Path(data.get('fixture_directory', ''))
        require(supplied_fixture.is_absolute() and supplied_fixture == expected_fixture and
                not supplied_fixture.is_symlink(), 'exact research-root child fixture directory required')
        fixture = inside_root(supplied_fixture)
        require(fixture.parent == ROOT and fixture == expected_fixture,
                'fixture directory cannot resolve through an alias or outside direct research root')
    else:
        require(int(rev[1:]) <= 5, 'Core>=v6 requires explicit reviewed fixture identity')
        fixture = ROOT / ('glm53-main-verified-source-engine-' + rev + '-prototype')
    require(isinstance(data.get('Core'), dict) and isinstance(data.get('fixture'), dict),
            'full Core and fixture source inventories required')
    require(read_json(fixture / 'CORE_SOURCE_PINS.json') == data['Core'],
            'fixture private headers must pin the exact receipt Core inventory')
    fixture_source = read_json(fixture / 'SOURCE_RECEIPT.json')
    require(fixture_source.get('new_core') == source.name and
            data['fixture'] == {**fixture_source['prototype_sources'],
                                'SOURCE_RECEIPT.json': sha(fixture / 'SOURCE_RECEIPT.json')},
            'full frozen fixture source receipt/inventory binding')
    for folder, expected in ((source, data['Core']), (fixture, data['fixture'])):
        require(not any(item.is_symlink() for item in folder.rglob('*')), 'no source inventory symlink')
        require({str(item.relative_to(folder)) for item in folder.rglob('*') if item.is_file()} == set(expected),
                'full source inventory file-set closure')
    for relative, expected in data['Core'].items():
        require(sha(source / relative) == expected, 'frozen Core drift: ' + relative)
    for relative, expected in data['fixture'].items():
        require(sha(fixture / relative) == expected, 'frozen fixture drift: ' + relative)
    for relative, expected in data['artifacts_sha256'].items():
        require(sha(ROOT / relative) == expected, 'matching built artifact drift: ' + relative)
    build = inside_root(data['Core_build'])
    checker = inside_root(data['fixture_build'])
    cache = (build / 'CMakeCache.txt').read_text()
    found = re.search(r'^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$', cache, re.M)
    require(found and Path(found.group(1)).resolve() == source.resolve(), 'archive/header source-home mismatch')
    manual = checker / 'glm53_main_verified_engine_manual'
    architecture = build / 'research-glm-tp4-home/architectures/glm5next_cache.so'
    for artifact in (manual, architecture):
        key = str(artifact.relative_to(ROOT))
        require(data['artifacts_sha256'].get(key) == sha(artifact), 'required artifact absent from build receipt')
    return data, source, build, manual, architecture


def verify_home(path):
    data = read_json(path)
    require(data.get('schema') == 'glm53-manual-research-home-v1' and data.get('passed') is True,
            'successful exact research HOME receipt')
    root = inside_root(data['home'])
    files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    dirs = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_dir()}
    require(files == set(data['files']) and dirs == {'architectures', 'kernels', 'quantizers'},
            'exclusive six-library HOME closure')
    require(len(files) == 6 and not any(p.is_symlink() for p in root.rglob('*')), 'private real copies only')
    for name, expected in data['files'].items():
        require(sha(root / name) == expected, 'research HOME drift: ' + name)
    require(sha(data['aql_build_receipt']) == data['aql_build_receipt_sha256'], 'AQL build receipt drift')
    return data, root


def prepare(args):
    source_receipt = verify_source()
    host, _, host_build, _, _ = verify_build(args.host_build_receipt, 'host')
    aql, _, _, manual, architecture = verify_build(args.aql_build_receipt, 'aql', host['revision'])
    out = Path(args.output).resolve(strict=False)
    require(out.parent == ROOT and not out.exists(), 'new immediate research child output directory')
    out.mkdir(mode=0o700)
    record = dict(schema='glm53-manual-preparation-v1', passed=False, commands=[], started=now(),
                  revision=host['revision'], production_changes=False, GPU_kernel_execution=False,
                  model_payload_read=False, source_receipt=str(source_receipt),
                  source_receipt_sha256=sha(source_receipt))
    save = lambda: atomic_json(out / 'receipt.json', record)
    save()
    try:
        download = read_json(ROOT / 'GLM53_CHECKPOINT_DOWNLOAD_STATUS.json')
        require(download['status'] == 'complete' and download['revision'] == 'b5688f25491202978c19c4d036eef579f61bbe07',
                'completed pinned AMD checkpoint metadata')
        model = Path(download['destination']).resolve(strict=True)
        reviewed = read_json(ROOT / 'glm53-manual-tokenizer-home-preflight-20261025/METADATA_REVIEW.json')
        for name, expected in reviewed['metadata_files'].items():
            require(sha(model / name) == expected['sha256'], 'reviewed tokenizer metadata drift: ' + name)
        prompt = out / 'prompt-a.txt'
        prompt.write_bytes(b'a')
        require(prompt.read_bytes() == b'a', 'one-byte no-newline prompt')
        with lock_file(ROOT / 'research-compiler.lock', timeout=7200):
            command = ['nice', '-n19', '/usr/bin/python3', str(ROOT / 'run_guarded_heavy_research.py'),
                       '--', '/usr/bin/cmake', '--build', str(host_build),
                       '--target', 'rad-tokcheck', '--parallel', '1']
            require(run_logged(command, out / 'tokenizer-build.log', 3600, safe_environment(), record, save) == 0,
                    'matching HOST tokenizer tool build')
        # Building the tool must not silently replace the successfully pinned Core.
        verify_build(args.host_build_receipt, 'host', host['revision'])
        tokcheck = host_build / 'bin/rad-tokcheck'
        require(tokcheck.is_file(), 'matching existing tokenizer tool target')
        command = [str(tokcheck), '--user', '--turns', '0', '--repeat', '1',
                   str(model / 'tokenizer.json'), str(prompt)]
        require(run_logged(command, out / 'tokenizer.log', 300, safe_environment(), record, save) == 0,
                'exact-mode HOST tokenizer preflight')
        text = (out / 'tokenizer.log').read_text()
        require(re.search(r'\btokenizer\s+1 ids\b', text) and 'reference  identical' in text,
                'one-token count and actual reference comparison required')
        home = out / 'exclusive-home'
        home.mkdir(mode=0o700)
        for name in ('architectures', 'kernels', 'quantizers'):
            (home / name).mkdir(mode=0o700)
        pins = read_json(ROOT / 'GLM53_SHIPPED_PLUGIN_HASHES.json')
        sources = {name: ROOT / 'shipped-radiance-1.12.0-home' / name for name in pins
                   if name.startswith(('kernels/', 'quantizers/'))}
        require(len(sources) == 5, 'exact pinned four kernels and one quantizer')
        files = {}
        for name, source in sources.items():
            require(sha(source) == pins[name], 'pinned shipped library drift: ' + name)
            shutil.copyfile(source, home / name)
            require(sha(home / name) == pins[name], 'private library copy mismatch')
            (home / name).chmod(0o444)
            files[name] = pins[name]
        name = 'architectures/glm5next_cache.so'
        files[name] = sha(architecture)
        shutil.copyfile(architecture, home / name)
        (home / name).chmod(0o444)
        require(sha(home / name) == files[name], 'matching experimental module copy')
        home_record = dict(schema='glm53-manual-research-home-v1', passed=True,
                           home=str(home), files=files, revision=aql['revision'],
                           aql_build_receipt=str(Path(args.aql_build_receipt).resolve()),
                           aql_build_receipt_sha256=sha(args.aql_build_receipt),
                           source_architecture=str(architecture),
                           shipped_pin_manifest_sha256=sha(ROOT / 'GLM53_SHIPPED_PLUGIN_HASHES.json'),
                           production_changes=False, GPU_kernel_execution=False)
        atomic_json(out / 'home-receipt.json', home_record)
        verify_home(out / 'home-receipt.json')
        record.update(host_build_receipt=str(Path(args.host_build_receipt).resolve()),
                      host_build_receipt_sha256=sha(args.host_build_receipt),
                      aql_build_receipt=str(Path(args.aql_build_receipt).resolve()),
                      aql_build_receipt_sha256=sha(args.aql_build_receipt),
                      manual_binary=str(manual), manual_binary_sha256=sha(manual),
                      tokenizer_binary=str(tokcheck), tokenizer_binary_sha256=sha(tokcheck),
                      tokenizer_metadata={name: expected['sha256'] for name, expected in reviewed['metadata_files'].items()},
                      model=str(model), prompt_file=str(prompt), prompt_sha256=sha(prompt), prompt_hex='61',
                      input_token_count_runtime=1, expected_id64_scope='static vocab metadata, not printed by rad-tokcheck',
                      home_receipt=str(out / 'home-receipt.json'), home_receipt_sha256=sha(out / 'home-receipt.json'),
                      passed=True)
    except BaseException as error:
        record['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        record['finished'] = now()
        record['logs_sha256'] = {p.name: sha(p) for p in out.glob('*.log')}
        save()
        print(out / 'receipt.json', flush=True)


def health():
    with urllib.request.urlopen(HEALTH, timeout=10) as response:
        require(response.status == 200, 'QFN health HTTP200')
        raw = response.read(65537)
        require(len(raw) <= 65536, 'bounded health reply')
    data = json.loads(raw)
    require(data.get('status') == 'ok' and isinstance(data.get('in_flight'), int), 'actual healthy QFN reply')
    return {'status': data['status'], 'in_flight': data['in_flight']}


def docker_capture(prefix, arguments, timeout=30):
    # INSPECT_FORMAT contains only safe selected state scalars. Config/Env,
    # commands, labels, mounts and health-log contents are NEVER requested.
    check_cancelled()
    proc = subprocess.Popen([*prefix, *arguments], cwd=ROOT, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        require(len(stdout) <= 65536 and len(stderr) <= 65536, 'bounded Docker reply')
        require(proc.returncode == 0, 'Docker ' + arguments[0] + ' failed: ' + stderr.decode(errors='replace')[:1000])
        return stdout
    finally:
        stop_owned(proc)


def inspect(prefix, reference):
    value = json.loads(docker_capture(prefix, ['inspect', '--type', 'container', '--format', INSPECT_FORMAT, reference]))
    require(set(value) == {'Id', 'Image', 'Name', 'Running', 'Status', 'Paused', 'Restarting', 'Dead',
                           'ExitCode', 'StartedAt', 'FinishedAt'}, 'safe inspect schema')
    require(re.fullmatch('[0-9a-f]{64}', value['Id']) and re.fullmatch('sha256:[0-9a-f]{64}', value['Image']),
            'immutable full container/image identities')
    return value


def same_identity(actual, original):
    require(all(actual[k] == original[k] for k in ('Id', 'Image', 'Name')) and original['Name'] == '/' + CONTAINER,
            'same original QFN container ID/image/name required')


def stopped(prefix, original):
    current = inspect(prefix, original['Id'])
    same_identity(current, original)
    require(current['Running'] is False and current['Status'] == 'exited' and
            not current['Paused'] and not current['Restarting'] and not current['Dead'],
            'original QFN must remain fully stopped before/during four-GPU child')
    return current


def bounded_ascii(path, limit):
    with Path(path).open('rb') as handle:
        raw = handle.read(limit + 1)
    require(len(raw) <= limit, 'bounded physical resource field: ' + str(path))
    return raw.decode('ascii').strip()


def physical_resources():
    # Passive sysfs/proc reads only: no device API, GPU query subprocess,
    # process inspection, reset, driver change or memory-management action.
    devices = []
    for card in sorted(Path('/sys/class/drm').glob('card*')):
        if not re.fullmatch(r'card[0-9]+', card.name):
            continue
        device = card / 'device'
        if not (device / 'vendor').is_file():
            continue  # non-PCI DRM nodes have no AMD vendor authority
        vendor = bounded_ascii(device / 'vendor', 32)
        require(re.fullmatch(r'0x[0-9a-fA-F]{4}', vendor), 'canonical PCI vendor field')
        if int(vendor, 16) != 0x1002:
            continue
        canonical = device.resolve(strict=True)
        bdf = canonical.name
        require(re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', bdf), 'canonical PCI BDF')
        values = []
        for leaf in ('mem_info_vram_total', 'mem_info_vram_used'):
            raw = bounded_ascii(device / leaf, 64)
            require(re.fullmatch(r'[0-9]{1,20}', raw), 'exact integer byte field: ' + leaf)
            values.append(int(raw))
        total, used = values
        require(0 <= used <= total, 'physical VRAM used/total bounds')
        devices.append(dict(card=card.name, bdf=bdf, vendor=vendor,
                            device_directory=str(canonical), total_bytes=total, used_bytes=used))
    require(len(devices) <= 16, 'bounded AMD device metadata')
    meminfo = bounded_ascii('/proc/meminfo', 16384)
    available = [line.split() for line in meminfo.splitlines() if line.startswith('MemAvailable:')]
    require(len(available) == 1 and len(available[0]) == 3 and available[0][1].isdecimal() and
            available[0][2] == 'kB', 'exact Linux MemAvailable kB field')
    available_bytes = int(available[0][1]) * 1024
    exact_devices = (len(devices) == 4 and {d['bdf'] for d in devices} == EXPECTED_AMD_BDFS and
                     len({d['device_directory'] for d in devices}) == 4 and
                     all(d['total_bytes'] == EXPECTED_VRAM_BYTES for d in devices))
    idle = exact_devices and all(d['used_bytes'] <= MAX_IDLE_VRAM_BYTES for d in devices)
    return dict(utc=now(), devices=devices, MemAvailable_bytes=available_bytes,
                exact_devices=exact_devices, all_vram_idle=idle,
                passed=idle and available_bytes >= MIN_AVAILABLE_BYTES)


def await_physical_resources(prefix, original, receipt, save):
    gate = dict(passed=False, started=now(), wait_seconds=RESOURCE_WAIT_SECONDS,
                min_MemAvailable_bytes=MIN_AVAILABLE_BYTES, max_vram_used_bytes_per_device=MAX_IDLE_VRAM_BYTES,
                expected_vram_total_bytes_per_device=EXPECTED_VRAM_BYTES,
                expected_AMD_BDFs=sorted(EXPECTED_AMD_BDFS), samples=[], GPU_query_subprocess=False)
    receipt['physical_resource_gate'] = gate
    deadline = time.monotonic() + RESOURCE_WAIT_SECONDS
    try:
        while True:
            check_cancelled()
            require(time.monotonic() < deadline, 'post-stop physical resource deadline')
            # A changed/running original container is an immediate refusal,
            # rather than a reason to wait while launching another GPU user.
            stopped(prefix, original)
            try:
                sample = physical_resources()
            except (OSError, RuntimeError, ValueError, UnicodeError) as error:
                sample = dict(utc=now(), passed=False, error=type(error).__name__ + ': ' + str(error))
            gate['samples'].append(sample)
            require(len(gate['samples']) <= 31, 'bounded physical resource samples')
            save()
            if sample['passed']:
                require(time.monotonic() < deadline, 'fresh resource sample within deadline')
                gate['passed'] = True
                gate['finished'] = now()
                save()
                return
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'post-stop VRAM/RAM did not reach admitted resource floors')
            time.sleep(min(2, remaining))
    finally:
        gate.setdefault('finished', now())
        save()


def verify_window_inputs(args):
    source_receipt = verify_source()
    prep_path = inside_root(args.preparation_receipt)
    prep = read_json(prep_path)
    require(prep.get('schema') == 'glm53-manual-preparation-v1' and prep.get('passed') is True,
            'successful prepared tokenizer/HOME receipt')
    revision = prep['revision']
    require(prep['source_receipt_sha256'] == sha(source_receipt), 'same reviewed runner source receipt')
    for key in ('host_build_receipt', 'aql_build_receipt', 'home_receipt'):
        require(sha(prep[key]) == prep[key + '_sha256'], 'prepared receipt drift: ' + key)
    host, _, _, _, _ = verify_build(prep['host_build_receipt'], 'host', revision)
    aql, _, _, binary, _ = verify_build(prep['aql_build_receipt'], 'aql', revision)
    home, home_path = verify_home(prep['home_receipt'])
    require(home['aql_build_receipt_sha256'] == prep['aql_build_receipt_sha256'], 'same AQL build/HOME')
    require(str(binary) == prep['manual_binary'] and sha(binary) == prep['manual_binary_sha256'], 'actual manual binary hash')
    require(sha(prep['prompt_file']) == prep['prompt_sha256'] and Path(prep['prompt_file']).read_bytes() == b'a', 'one-byte prompt provenance')
    require(sha(prep['tokenizer_binary']) == prep['tokenizer_binary_sha256'] and prep['input_token_count_runtime'] == 1,
            'actual tokenizer preflight binding')
    model = Path(prep['model'])
    for name, expected in prep['tokenizer_metadata'].items():
        require(sha(model / name) == expected, 'actual tokenizer metadata changed')
    require(len(args.host_validation) == 3, 'Release/ASan/TSan HOST validation receipts required')
    modes = set()
    for path in args.host_validation:
        data = read_json(inside_root(path))
        require(data.get('passed') is True and data.get('revision') == revision and
                data.get('mode') in ('release', 'asan', 'tsan'), 'matching successful HOST regression mode')
        modes.add(data['mode'])
        require(data.get('GPU_execution') is False and data.get('production_changes') is False, 'HOST gate scope')
        if data['mode'] == 'release':
            require(data['build_receipt_sha256'] == prep['host_build_receipt_sha256'], 'same HOST Release build validated')
        require(len(data.get('commands', [])) >= 20 and all(row.get('returncode') == 0 for row in data['commands']),
                'completed actual HOST regression commands required')
        require(bool(data.get('runtime_artifacts_sha256')), 'nonempty HOST executable/plugin artifacts')
        candidates = ROOT.glob('build-glm53-main-verified-engine-' + revision + '-host-' + data['mode'] + '-evidence-*/receipt.json')
        matched = [candidate for candidate in candidates if sha(candidate) == data['build_receipt_sha256']]
        require(len(matched) == 1, 'unique actual HOST build receipt bound by validation digest')
        verify_build(matched[0], 'host', revision, mode=data['mode'])
        for relative, expected in data['runtime_artifacts_sha256'].items():
            require(sha(ROOT / relative) == expected, 'validated HOST runtime artifact drift')
    require(modes == {'release', 'asan', 'tsan'}, 'all three distinct HOST modes')
    assembly_path = inside_root(args.assembly_receipt)
    assembly = read_json(assembly_path)
    require(assembly.get('passed') is True and assembly.get('complete_main_expert_store_packed') is True, 'actual full assembly PASS')
    domain = assembly['complete_result']
    require(domain['passed'] is True and domain['records'] == RECORDS and domain['shards'] == 168 and
            domain['native_bytes'] == NATIVE_BYTES and domain['bundle_id'] == BUNDLE_ID and domain['mode'] == 'complete_main_data',
            'exact actual complete domain/common bundle')
    bundle = assembly_path.parent / 'private-complete-store/data'
    require(sha(assembly_path.parent / 'private-complete-store/receipt.json') == assembly['complete_result_sha256'], 'actual assembly result drift')
    proof = Path(assembly['proof_receipt']).parent / 'complete-source-proof.json'
    require(sha(assembly['proof_receipt']) == assembly['proof_receipt_sha256'] and
            sha(proof) == assembly['proof_sha256'] and sha(bundle / 'source-proof.json') == assembly['proof_sha256'],
            'same immutable producer proof')
    source_path = inside_root(args.source_pass_receipt)
    source = read_json(source_path)
    require(source.get('passed') is True and source.get('assembly_receipt_sha256') == sha(assembly_path), 'actual complete reader/provider PASS')
    result = source['result']
    require(result.get('passed') is True and result['bundle_id'] == BUNDLE_ID and
            result['complete_native_bytes_verified'] == NATIVE_BYTES and result['raw_tensors'] == 76108 and
            len(result['rank_reports']) == 4, 'actual same complete reader result')
    for rank, row in enumerate(result['rank_reports']):
        require(row['rank'] == rank and row['records'] == 12096 and row['source_cut_bytes_hashed'] == NATIVE_BYTES // 4 and
                row['bundle_id'] == BUNDLE_ID, 'all four source-cut authorities')
    prewalk = read_json(inside_root(args.prewalk_receipt))
    require(prewalk.get('schema') == 'glm53-verified-component-run-v1' and prewalk.get('passed') is True and
            prewalk.get('mode') == 'complete-prewalk' and prewalk.get('GPU_kernel_execution') is False and
            prewalk.get('production_changes') is False, 'successful actual source prewalk gate schema')
    require(prewalk['core_build_receipt_sha256'] == prep['aql_build_receipt_sha256'] and
            prewalk['home_receipt_sha256'] == prep['home_receipt_sha256'] and
            prewalk['assembly_receipt_sha256'] == sha(assembly_path) and
            prewalk['source_pass_receipt_sha256'] == sha(source_path), 'prewalk bound to same Core/HOME/data receipts')
    require(sha(prewalk['core_build_receipt']) == prewalk['core_build_receipt_sha256'] and
            sha(prewalk['component_build_receipt']) == prewalk['component_build_receipt_sha256'] and
            sha(prewalk['result_file']) == prewalk['result_sha256'], 'prewalk artifacts unchanged')
    component = read_json(prewalk['component_build_receipt'])
    require(component.get('passed') is True and component.get('backend') == 'aql' and
            component.get('core_build_receipt_sha256') == prep['aql_build_receipt_sha256'], 'same successfully built component checker')
    component_binary = Path(component['build']) / 'glm53_main_verified_graph_quotes'
    require(sha(component_binary) == component['binary_sha256']['glm53_main_verified_graph_quotes'], 'actual prewalk checker binary')
    actual = read_json(prewalk['result_file'])
    quote = actual['actual_complete_data_prewalk']
    require(actual.get('passed') is True and actual.get('runtime_ready') is False and
            actual.get('kernel_instances_or_launches') == 0 and actual.get('actual_logical_rank_graphs') == 4 and
            actual.get('runtime_SourceFactory_or_Session_or_Engine_admitted') is False and
            quote['complete_bundle_id'] == BUNDLE_ID and quote['verified_native_bytes'] == NATIVE_BYTES and
            quote['source_rank_cut_hash_bytes'] == NATIVE_BYTES and quote['omitted_views'] == 193536 and
            quote['omitted_source_bytes'] == NATIVE_BYTES and quote['loader_execute_called'] is False and
            quote['jobs'] > 0 and quote['max_input_bytes'] <= (2 << 30) and quote['max_nonidentity_output_bytes'] <= (2 << 30) and
            quote['column_scratch_bytes'] <= (64 << 20) and actual.get('refusals', 0) >= 15,
            'complete actual graph/Sampler/ordinary-loader prewalk/cap refusals')
    return prep, home_path, model, bundle, proof, binary, quote


def final_json(path):
    require(Path(path).stat().st_size <= MAX_LOG, 'bounded combined output')
    text = Path(path).read_text(errors='replace')
    starts = [0] if text.startswith('{') else []
    starts += [m.start() + 1 for m in re.finditer(r'\n\{', text)]
    for at in reversed(starts):
        try:
            value = json.loads(text[at:])
            if value.get('schema') == 'glm53-actual-verified-main-engine-v1':
                return value
        except (ValueError, AttributeError):
            pass
    raise RuntimeError('combined child did not publish its actual final Engine result')


def restore(prefix, original, ambiguous, receipt, save):
    global _RESTORING
    _RESTORING = True
    result = dict(attempted=True, started=now(), same_container_id=original['Id'],
                  original_stop_ambiguous=ambiguous, passed=False)
    receipt['restoration'] = result

    def evidence():
        # Evidence storage failure must NEVER prevent physical QFN restoration.
        try:
            save()
        except BaseException as error:
            message = type(error).__name__ + ': ' + str(error)
            result['evidence_write_error'] = message
            try:
                print('Restoration evidence unavailable: ' + message, file=sys.stderr, flush=True)
            except BaseException:
                pass

    evidence()
    deadline = time.monotonic() + RESTORE_SECONDS
    boundary = not ambiguous  # acknowledged stop already passed stopped(...)
    stable_since = None
    start_requested = False
    last_error = None
    try:
        while time.monotonic() < deadline:
            try:
                current = inspect(prefix, original['Id'])
                same_identity(current, original)
                if not boundary:
                    # A killed stop client may leave daemon work in flight.
                    # Never accept an early healthy sample as proof of recovery.
                    if (current['Running'] is False and current['Status'] == 'exited' and
                            not current['Paused'] and not current['Restarting'] and not current['Dead']):
                        boundary = True
                        result['recovery_stop_boundary'] = True
                        result['recovery_stopped_state'] = current
                        evidence()
                    else:
                        # Even a timed-out/negative repeat stop can complete in
                        # the daemon. Re-inspect on the next loop in either case.
                        try:
                            remaining = max(1, deadline - time.monotonic())
                            docker_capture(prefix, ['stop', '-t', str(STOP_SECONDS), original['Id']],
                                           min(STOP_CLIENT_SECONDS, remaining))
                        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                            result['recovery_stop_error'] = type(error).__name__ + ': ' + str(error)
                        evidence()
                        time.sleep(2)
                        continue
                if not current['Running']:
                    require(current['Status'] == 'exited' and not current['Paused'] and
                            not current['Restarting'] and not current['Dead'], 'restart only fully exited original')
                    stable_since = None
                    remaining = max(1, deadline - time.monotonic())
                    docker_capture(prefix, ['start', original['Id']], min(120, remaining))
                    start_requested = True
                    result['start_requested'] = True
                    evidence()
                    time.sleep(2)
                    continue
                require(current['Status'] == 'running' and not current['Paused'] and not current['Restarting'] and
                        not current['Dead'] and current['StartedAt'] != original['StartedAt'],
                        'new successful original-container start after stopped boundary')
                good = health()
                if stable_since is None:
                    stable_since = time.monotonic()
                result.update(last_state=current, health=good, stable_seconds=time.monotonic() - stable_since)
                evidence()
                if time.monotonic() - stable_since >= 40:
                    # A stop with missing acknowledgement remains explicitly
                    # unproven overall, despite successful best-effort recovery.
                    result['passed'] = not ambiguous
                    result['healthy_after_best_effort'] = True
                    result['start_requested'] = start_requested
                    if ambiguous:
                        result['error'] = 'original stop acknowledgement ambiguous; recovery is best effort'
                    return result['passed']
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                last_error = type(error).__name__ + ': ' + str(error)
                stable_since = None
                result['last_error'] = last_error
                evidence()
            time.sleep(2)
        raise RuntimeError('same-container restoration/health deadline: ' + str(last_error))
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        return False
    finally:
        result['finished'] = now()
        evidence()


def run_window(args):
    # All potentially slow artifact verification occurs while QFN is still up.
    prepared = verify_window_inputs(args)
    out = Path(args.output).resolve(strict=False)
    require(out.parent == ROOT and not out.exists(), 'new immediate research child evidence directory')
    out.mkdir(mode=0o700)
    receipt = dict(schema='glm53-controlled-manual-maintenance-v1', passed=False, commands=[], started=now(),
                   scope='one controlled stop/start of the SAME existing QFN container plus one combined isolated GLM startup+one-token child',
                   configuration_or_deployment_changes=False, separate_independent_transport_run=False,
                   source_receipt=str(verify_source()), source_receipt_sha256=sha(verify_source()),
                   preparation_receipt_sha256=sha(args.preparation_receipt),
                   assembly_receipt_sha256=sha(args.assembly_receipt), source_pass_receipt_sha256=sha(args.source_pass_receipt),
                   prewalk_receipt_sha256=sha(args.prewalk_receipt),
                   host_validation_receipts={str(Path(p).resolve()): sha(p) for p in args.host_validation},
                   external_SIGKILL_or_power_loss_recovery_guaranteed=False)
    save = lambda: atomic_json(out / 'receipt.json', receipt)
    save()
    prefix = ['/usr/bin/sudo', '-n', '/usr/bin/docker'] if args.sudo_docker else ['/usr/bin/docker']
    original = None
    armed = False
    stop_confirmed = False
    combined = False
    restored = False
    try:
        with lock_file(ROOT / 'research-heavy-runtime.lock', timeout=180):
            try:
                # Revalidate after waiting for the lock, immediately before the
                # fresh health/stop boundary. No stale copied executable/home.
                prep, home, model, bundle, proof, binary, quote = verify_window_inputs(args)
                original = inspect(prefix, CONTAINER)
                require(original['Name'] == '/' + CONTAINER and original['Running'] is True and
                        original['Status'] == 'running' and not original['Paused'] and not original['Restarting'] and not original['Dead'],
                        'existing healthy-running original QFN identity')
                receipt['original_container'] = original
                first = health();require(first['in_flight'] == 0, 'QFN must be idle before authorized maintenance')
                time.sleep(1)
                second = health();require(second['in_flight'] == 0, 'fresh QFN idle confirmation')
                receipt['health_before'] = [first, second]
                check_cancelled()
                armed = True  # BEFORE stop: timeout/nonzero can still have stopped daemon container
                receipt['restoration_armed_before_stop'] = True
                save()
                docker_capture(prefix, ['stop', '-t', str(STOP_SECONDS), original['Id']], STOP_CLIENT_SECONDS)
                receipt['stopped_state'] = stopped(prefix, original)
                stop_confirmed = True
                save()
                check_cancelled()
                # Hash all checker/source/receipts immediately before the sole
                # four-GPU child as well, while captured original ID stays stopped.
                prep, home, model, bundle, proof, binary, quote = verify_window_inputs(args)
                await_physical_resources(prefix, original, receipt, save)
                command = ['nice', '-n19', '/usr/bin/ionice', '-c', '3', str(binary), '--one-token',
                           str(model), str(home), str(bundle), str(proof), str(BUNDLE_ID), 'a']
                receipt['GPU_visibility'] = {'ROCR_VISIBLE_DEVICES': '0,1,2,3', 'HIP_VISIBLE_DEVICES': '0,1,2,3'}
                receipt['combined_child_attempted'] = True
                rc = run_logged(command, out / 'combined-engine.log', args.test_timeout, safe_environment(True), receipt, save,
                                guard=lambda: stopped(prefix, original))
                require(rc == 0, 'combined actual GLM child failed with exit ' + str(rc))
                actual = final_json(out / 'combined-engine.log')
                receipt['combined_result'] = actual
                require(actual.get('passed') is True and actual.get('status') == 0 and actual.get('source_admitted') is True and
                        actual.get('transport_ready') is True and actual.get('text_complete') is True and
                        actual.get('bundle_id') == BUNDLE_ID and actual.get('native_bytes_verified') == NATIVE_BYTES and
                        actual.get('raw_mapped_bytes_compared') == 185066521464 and actual.get('max_tok') == 1 and actual.get('max_seqs') == 1 and
                        actual.get('max_spec') == 0 and actual.get('tp_wire') == 'exact' and actual.get('fixed_residents') == 0 and
                        isinstance(actual.get('sampled_token'), int) and 0 <= actual['sampled_token'] < 154880 and
                        actual.get('ordinary_jobs') == quote['jobs'] and actual.get('omitted_expert_views') == 193536 and
                        actual.get('ordinary_upload_bytes') == quote['ordinary_stored_bytes'] and
                        actual.get('ordinary_successful_drains') == quote['jobs'], 'actual combined startup/text/ordinary result closure')
                # A zero exit after the final JSON follows the current Core's
                # explicit healthy teardown; known retirement failures exit88.
                receipt['combined_startup_plus_text_passed'] = True
                combined = True
            finally:
                if armed and original is not None:
                    restored = restore(prefix, original, not stop_confirmed, receipt, save)
                else:
                    receipt['restoration'] = {'attempted': False, 'reason': 'stop was never armed/issued'}
                    restored = True
                save()
        receipt['passed'] = combined and restored and not _INTERRUPTED
    except BaseException as error:
        receipt['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        receipt.update(finished=now(), stop_boundary_confirmed=stop_confirmed,
                       combined_startup_plus_text_passed=combined,
                       same_original_container_restored=restored if armed else None,
                       logs_sha256={p.name: sha(p) for p in out.glob('*.log')})
        save()
        print(out / 'receipt.json', flush=True)
    return 0 if receipt['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    prep = commands.add_parser('prepare', help='HOST tokenizer and six-library private HOME only; no Docker operation')
    prep.add_argument('--host-build-receipt', type=Path, required=True)
    prep.add_argument('--aql-build-receipt', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run-window', help='explicit authorized single maintenance window; NEVER a build/install action')
    run.add_argument('--preparation-receipt', type=Path, required=True)
    run.add_argument('--assembly-receipt', type=Path, required=True)
    run.add_argument('--source-pass-receipt', type=Path, required=True)
    run.add_argument('--prewalk-receipt', type=Path, required=True)
    run.add_argument('--host-validation', type=Path, action='append', required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--test-timeout', type=int, default=3600)
    run.add_argument('--sudo-docker', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.mode == 'run-window':
        require(60 <= args.test_timeout <= 7200, 'explicit bounded combined child timeout')
    install_signals()
    if args.mode == 'prepare':
        prepare(args)
        return 0
    return run_window(args)


if __name__ == '__main__':
    sys.exit(main())
