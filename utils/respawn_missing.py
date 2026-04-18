#!/usr/bin/env python3
"""Additive, idempotent spawner for the CW2 Gazebo world.

Flow:
  1. Wait for /clock (Gazebo alive).
  2. Wait for any `ros_gz_sim create` stragglers from launch_simulation.py to
     finish; pkill the tail if they don't exit by themselves.
  3. Wait for `ign model --list` to return cleanly — proof the server isn't
     backed up on the create queue.
  4. Diff world.yaml against the live model list.
  5. Spawn missing models serially with a short gap, verifying each via
     `ign model --list` and retrying once on failure.
  6. Never delete anything — preserves plugin state and avoids the
     delete-mid-PreUpdate segfault that crashed Gazebo in the old flow.
"""

import os
import subprocess
import sys
import time
import yaml


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def wait_for_clock(timeout=120.0):
    print('[respawn] Waiting for Gazebo /clock...', flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = run(['ros2', 'topic', 'list'], timeout=10)
        except subprocess.TimeoutExpired:
            time.sleep(2.0)
            continue
        if '/clock' in r.stdout:
            print('[respawn] /clock detected.', flush=True)
            return True
        time.sleep(2.0)
    print('[respawn] Timed out waiting for /clock.', flush=True)
    return False


def wait_for_create_stragglers(timeout=60.0):
    """Wait for ros_gz_sim create processes from launch_simulation.py to
    exit. pkill anything still alive at the end — they're stuck."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = run(['pgrep', '-f', 'ros_gz_sim.*create'])
        if not r.stdout.strip():
            print('[respawn] No create stragglers.', flush=True)
            return
        time.sleep(2.0)
    n_left = len(run(['pgrep', '-f', 'ros_gz_sim.*create']).stdout.split())
    print(f'[respawn] {n_left} create process(es) still alive after {timeout}s — killing.', flush=True)
    run(['pkill', '-9', '-f', 'ros_gz_sim.*create'])
    time.sleep(1.0)


def list_models(timeout=10.0):
    """Authoritative top-level model list from the running Gazebo. Returns
    None if the server is unresponsive or timed out."""
    try:
        r = run(['ign', 'model', '--list'], timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if 'timed out' in r.stdout.lower():
        return None
    models = set()
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith('- '):
            models.add(s[2:])
    return models


def wait_for_server_idle(timeout=180.0):
    """Poll ign model --list until it responds — proof the server is alive
    and not backed up on the create queue. Falls through to one
    last-chance attempt with a generous timeout before giving up."""
    print(f'[respawn] Waiting for Gazebo server to be idle (up to {int(timeout)}s)...', flush=True)
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        m = list_models(timeout=8.0)
        if m is not None:
            print(f'[respawn] Server idle after {attempt} attempt(s). '
                  f'{len(m)} model(s) present.', flush=True)
            return True
        time.sleep(1.0)
    print('[respawn] Idle wait exhausted — one last-chance list (20s timeout)...', flush=True)
    m = list_models(timeout=20.0)
    if m is not None:
        print(f'[respawn] Last-chance list succeeded. {len(m)} model(s) present.', flush=True)
        return True
    print('[respawn] Server never became responsive.', flush=True)
    return False


def spawn_model(world_name, name, sdf_file, x, y, z):
    cmd = [
        'ros2', 'run', 'ros_gz_sim', 'create',
        '-world', world_name,
        '-file', sdf_file,
        '-name', name,
        '-allow_renaming', 'false',
        '-x', str(x), '-y', str(y), '-z', str(z),
    ]
    try:
        run(cmd, timeout=30)
    except subprocess.TimeoutExpired:
        pass  # caller re-verifies with list_models


def remove_model(world_name, name, timeout=10.0):
    """Best-effort removal of a single model via the gz service API.
    launch_simulation.py's parallel race occasionally loses and respawns
    a drone with `-allow_renaming=true`, leaving a stray `<name>_N`
    sitting next to the authoritative `<name>`. We remove those
    stragglers before computing `missing`, so they don't mask a genuine
    absence (and so nobody wonders why there's a ghost drone0_0 in rviz).

    Single-model deletion is safer than the old wholesale-wipe flow
    (which segfaulted Gazebo mid-PreUpdate) because:
      * the duplicate has no ROS-side nodes attached (platform, state
        estimator, etc. were only launched for the expected names);
      * it happens before takeoff, when controller loops haven't started;
      * failures are logged and don't abort respawn.
    """
    try:
        r = run([
            'ign', 'service',
            '-s', f'/world/{world_name}/remove',
            '--reqtype', 'ignition.msgs.Entity',
            '--reptype', 'ignition.msgs.Boolean',
            '--timeout', '5000',
            '--req', f'name: "{name}", type: MODEL',
        ], timeout=timeout)
        ok = r.returncode == 0 and 'data: true' in (r.stdout or '')
        if not ok:
            print(
                f'[respawn]   remove {name}: rc={r.returncode} '
                f'out={(r.stdout or "").strip()!r}',
                flush=True,
            )
    except subprocess.TimeoutExpired:
        print(f'[respawn]   remove {name} timed out', flush=True)


def find_rename_duplicates(current, expected):
    """Return names in `current` that look like `<e>_<N>` where `<e>`
    is an expected base name and `<e>` is also present in `current`.
    Strict: N must be pure digits, so we don't false-positive on names
    that legitimately end in an underscore suffix."""
    dupes = []
    for m in current:
        for e in expected:
            prefix = f'{e}_'
            if (
                m.startswith(prefix)
                and m[len(prefix):].isdigit()
                and e in current
            ):
                dupes.append(m)
                break
    return dupes


def find_object_sdf(model_type):
    """Search GZ_SIM_RESOURCE_PATH for <model_type>/<model_type>.sdf."""
    for p in os.environ.get('GZ_SIM_RESOURCE_PATH', '').split(':'):
        if not p:
            continue
        candidate = os.path.join(p, model_type, f'{model_type}.sdf')
        if os.path.isfile(candidate):
            return candidate
    return None


def build_expected_models(world):
    """From world.yaml, build {name: (sdf_path, (x,y,z))} for every
    model we expect to be in the world."""
    expected = {}

    drone_idx = 0
    for drone in world.get('drones', []):
        name = drone['model_name']
        model_type = drone['model_type']
        xyz = drone.get('xyz', [0.0, 0.0, 0.2])
        # launch_simulation.py materialises per-drone SDFs under /tmp
        sdf = f'/tmp/{model_type}_{drone_idx}.sdf'
        drone_idx += 1
        if os.path.isfile(sdf):
            expected[name] = (sdf, xyz)
        else:
            print(f'[respawn] WARN drone SDF not yet written: {sdf}', flush=True)

    for obj in world.get('objects', []):
        name = obj['model_name']
        model_type = obj['model_type']
        xyz = obj.get('xyz', [0.0, 0.0, 0.0])
        sdf = find_object_sdf(model_type)
        if sdf:
            expected[name] = (sdf, xyz)
        else:
            print(f'[respawn] WARN object SDF not found for {name} (type={model_type})', flush=True)

    return expected


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    world_yaml_path = os.path.join(
        project_dir, 'config_sim', 'world', 'world.yaml')

    if not os.path.exists(world_yaml_path):
        print(f'[respawn] world.yaml not found: {world_yaml_path}', flush=True)
        sys.exit(1)

    with open(world_yaml_path) as f:
        world = yaml.safe_load(f)
    world_name = world.get('world_name', 'empty')

    if not wait_for_clock(120.0):
        sys.exit(1)

    wait_for_create_stragglers(90.0)

    if not wait_for_server_idle(180.0):
        print('[respawn] Server unresponsive — aborting (would crash under load).', flush=True)
        sys.exit(1)

    expected = build_expected_models(world)
    if not expected:
        print('[respawn] No expected models — check world.yaml/SDF paths.', flush=True)
        sys.exit(1)

    current = list_models()
    if current is None:
        print('[respawn] Server went away after idle check — aborting.', flush=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Dedupe: kill any `<expected>_<N>` stragglers from launch_simulation.py's
    # parallel create race before diffing. Opt out with RESPAWN_DEDUPE=0.
    # ------------------------------------------------------------------
    if os.environ.get('RESPAWN_DEDUPE', '1') != '0':
        dupes = find_rename_duplicates(current, expected)
        if dupes:
            print(
                f'[respawn] Dedupe: removing {len(dupes)} renamed duplicate(s): {dupes}',
                flush=True,
            )
            for d in dupes:
                remove_model(world_name, d)
            time.sleep(1.0)
            refreshed = list_models()
            if refreshed is not None:
                current = refreshed

    missing = [n for n in expected if n not in current]
    print(f'[respawn] Expected {len(expected)}; {len(current)} present, {len(missing)} missing.', flush=True)
    print(f'[respawn] Missing: {missing}', flush=True)
    if not missing:
        print('[respawn] Nothing to do.', flush=True)
        return

    failed = []
    for name in missing:
        sdf, xyz = expected[name]
        print(f'[respawn] spawning {name} from {sdf} at {xyz} ...', flush=True)
        spawn_model(world_name, name, sdf, xyz[0], xyz[1], xyz[2])
        time.sleep(0.4)
        current = list_models() or set()
        if name in current:
            print(f'[respawn]   OK: {name}', flush=True)
            continue
        # First attempt didn't materialise. One retry with a longer pause.
        print(f'[respawn]   not visible after first try — retrying', flush=True)
        time.sleep(1.0)
        spawn_model(world_name, name, sdf, xyz[0], xyz[1], xyz[2])
        time.sleep(0.8)
        current = list_models() or set()
        if name in current:
            print(f'[respawn]   OK on retry: {name}', flush=True)
        else:
            print(f'[respawn]   FAILED: {name}', flush=True)
            failed.append(name)

    final = list_models() or set()
    print(f'[respawn] Final model count: {len(final)}. Failed: {failed}', flush=True)
    sys.exit(2 if failed else 0)


if __name__ == '__main__':
    main()
