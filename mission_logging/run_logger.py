from __future__ import annotations

import csv
import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Try common odometry message types used in ROS2.
try:
    from nav_msgs.msg import Odometry  # type: ignore
except Exception:  # pragma: no cover
    Odometry = None  # type: ignore

try:
    from geometry_msgs.msg import PoseStamped  # type: ignore
except Exception:  # pragma: no cover
    PoseStamped = None  # type: ignore


def _utc_iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _safe_name(s: str) -> str:
    # filesystem-safe-ish
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def _git_commit_short(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


@dataclass
class RunMeta:
    scenario_path: str
    drone_namespace: str
    planner: str
    use_sim_time: bool
    verbose: bool
    started_at_utc: str
    git_commit: Optional[str]
    extra: Dict[str, Any]


class RunLogger:
    """
    Auto-logs:
      - run_meta.json
      - events.jsonl
      - trajectory.csv (from odom)
      - metrics.json (computed at end)
      - stdout.log (tee)
    """

    def __init__(
        self,
        *,
        scenario_path: str,
        drone_namespace: str,
        planner: str = "baseline",
        use_sim_time: bool,
        verbose: bool,
        runs_root: str = "runs",
        odom_topic: Optional[str] = None,
        sample_hz: float = 10.0,
        repo_root: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.scenario_path = scenario_path
        self.drone_namespace = drone_namespace
        self.planner = planner
        self.use_sim_time = use_sim_time
        self.verbose = verbose
        self.sample_hz = sample_hz

        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.git_commit = _git_commit_short(self.repo_root)

        scen_name = _safe_name(Path(scenario_path).stem)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        run_dir_name = f"{stamp}__{scen_name}__{_safe_name(planner)}__{_safe_name(drone_namespace)}"
        self.run_dir = Path(runs_root) / run_dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Files
        self.meta_path = self.run_dir / "run_meta.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.traj_path = self.run_dir / "trajectory.csv"
        self.metrics_path = self.run_dir / "metrics.json"
        self.stdout_path = self.run_dir / "stdout.log"

        # Tee stdout/stderr
        self._stdout_tee = _Tee(self.stdout_path)
        self._stdout_tee.start()

        # ROS / telemetry
        self._node: Optional[Node] = None
        self._odom_lock = threading.Lock()
        self._latest_pose: Optional[Dict[str, float]] = None
        self._latest_stamp: Optional[float] = None

        # Derived trajectory for metrics
        self._traj_points: list[Dict[str, float]] = []
        self._traj_writer: Optional[csv.DictWriter] = None
        self._traj_fh = None

        # topic defaults (common in AS2): /<ns>/self_localization/odom OR /<ns>/odom
        if odom_topic is None:
            odom_topic = f"/{drone_namespace}/self_localization/pose"
        self.odom_topic = odom_topic

        # Timing
        self.t0_wall = time.time()
        self._closed = False

        meta = RunMeta(
            scenario_path=scenario_path,
            drone_namespace=drone_namespace,
            planner=planner,
            use_sim_time=use_sim_time,
            verbose=verbose,
            started_at_utc=_utc_iso(self.t0_wall),
            git_commit=self.git_commit,
            extra=extra_meta or {},
        )
        self._write_json(self.meta_path, dataclasses.asdict(meta))

        # init trajectory file header immediately
        self._traj_fh = open(self.traj_path, "w", newline="")
        self._traj_writer = csv.DictWriter(
            self._traj_fh,
            fieldnames=["t_wall", "t_rel", "x", "y", "z", "yaw", "speed_est"],
        )
        self._traj_writer.writeheader()
        self._traj_fh.flush()

        self.event("run_started", {"run_dir": str(self.run_dir), "odom_topic": self.odom_topic})

    @property
    def run_directory(self) -> Path:
        return self.run_dir

    def attach_ros(self, node: Node) -> None:
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

        self._node = node

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Discover the type actually published on this topic
        topic_map = dict(node.get_topic_names_and_types())
        topic_types = topic_map.get(self.odom_topic, [])
        self.event("telemetry_topic_types", {"topic": self.odom_topic, "types": topic_types})

        # Subscribe based on declared type
        if "geometry_msgs/msg/PoseStamped" in topic_types:
            if PoseStamped is None:
                self.event("telemetry_unavailable", {"reason": "PoseStamped not available in environment"})
                return
            node.create_subscription(PoseStamped, self.odom_topic, self._on_pose, qos)
            self.event("telemetry_subscribed", {"msg_type": "geometry_msgs/msg/PoseStamped", "topic": self.odom_topic})
            return

        if "nav_msgs/msg/Odometry" in topic_types:
            if Odometry is None:
                self.event("telemetry_unavailable", {"reason": "Odometry not available in environment"})
                return
            node.create_subscription(Odometry, self.odom_topic, self._on_odom, qos)
            self.event("telemetry_subscribed", {"msg_type": "nav_msgs/msg/Odometry", "topic": self.odom_topic})
            return

        # If nothing is publishing yet, fall back based on topic suffix
        if self.odom_topic.endswith("/pose"):
            if PoseStamped is not None:
                node.create_subscription(PoseStamped, self.odom_topic, self._on_pose, qos)
                self.event("telemetry_subscribed_fallback", {"msg_type": "geometry_msgs/msg/PoseStamped", "topic": self.odom_topic})
                return

        if Odometry is not None:
            node.create_subscription(Odometry, self.odom_topic, self._on_odom, qos)
            self.event("telemetry_subscribed_fallback", {"msg_type": "nav_msgs/msg/Odometry", "topic": self.odom_topic})
            return

        self.event("telemetry_unavailable", {"reason": "Could not subscribe (unknown type / missing msgs)"})
        return

    def latest_pose(self) -> Optional[Dict[str, float]]:
        with self._odom_lock:
            if self._latest_pose is None:
                return None
            return dict(self._latest_pose)

    def wait_for_first_pose(self, timeout_s: float = 5.0) -> Optional[Dict[str, float]]:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            pose = self.latest_pose()
            if pose is not None:
                return pose
            if self._node is not None:
                rclpy.spin_once(self._node, timeout_sec=0.05)
            else:
                time.sleep(0.05)
        return None

    def _on_odom(self, msg) -> None:
        # msg: nav_msgs/Odometry
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        now = time.time()
        with self._odom_lock:
            self._latest_pose = {"x": float(p.x), "y": float(p.y), "z": float(p.z), "yaw": float(yaw)}
            self._latest_stamp = now

    def _on_pose(self, msg) -> None:
        # msg: geometry_msgs/PoseStamped
        p = msg.pose.position
        q = msg.pose.orientation
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        now = time.time()
        with self._odom_lock:
            self._latest_pose = {"x": float(p.x), "y": float(p.y), "z": float(p.z), "yaw": float(yaw)}
            self._latest_stamp = now

    def _qos_sensor_data(self) -> QoSProfile:
        return QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

    def tick(self) -> None:
        """
        Call this regularly while mission is running to sample trajectory.
        (We keep it explicit so you control overhead.)
        """
        if self._closed:
            return

        pose = None
        with self._odom_lock:
            if self._latest_pose is not None:
                pose = dict(self._latest_pose)

        if pose is None:
            return

        t_wall = time.time()
        t_rel = t_wall - self.t0_wall

        # speed estimate from last point
        speed = 0.0
        if self._traj_points:
            prev = self._traj_points[-1]
            dt = max(1e-6, t_rel - prev["t_rel"])
            dx = pose["x"] - prev["x"]
            dy = pose["y"] - prev["y"]
            dz = pose["z"] - prev["z"]
            speed = (dx * dx + dy * dy + dz * dz) ** 0.5 / dt

        row = {
            "t_wall": t_wall,
            "t_rel": t_rel,
            "x": pose["x"],
            "y": pose["y"],
            "z": pose["z"],
            "yaw": pose["yaw"],
            "speed_est": speed,
        }

        self._traj_points.append(row)
        if self._traj_writer is not None:
            self._traj_writer.writerow(row)
            self._traj_fh.flush()

    def event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        rec = {
            "t_wall": time.time(),
            "t_rel": time.time() - self.t0_wall,
            "event": name,
            "payload": payload or {},
        }
        with open(self.events_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def finalize(self, *, success: bool, ended_reason: str = "normal") -> None:
        if self._closed:
            return
        self._closed = True

        t1 = time.time()
        self.event("run_finished", {"success": success, "ended_reason": ended_reason, "ended_at_utc": _utc_iso(t1)})

        metrics = self._compute_metrics(success=success, t_end=t1)
        self._write_json(self.metrics_path, metrics)

        if self._traj_fh is not None:
            self._traj_fh.close()

        self._stdout_tee.stop()

        # print a one-liner summary (also ends up in stdout.log)
        print(f"[RunLogger] Saved run to: {self.run_dir}")
        print(f"[RunLogger] Summary: {json.dumps(metrics, indent=2)}")

    def _compute_metrics(self, *, success: bool, t_end: float) -> Dict[str, Any]:
        duration = t_end - self.t0_wall

        # path length
        path_len = 0.0
        speeds = []
        if len(self._traj_points) >= 2:
            for a, b in zip(self._traj_points[:-1], self._traj_points[1:]):
                dx = b["x"] - a["x"]
                dy = b["y"] - a["y"]
                dz = b["z"] - a["z"]
                path_len += (dx * dx + dy * dy + dz * dz) ** 0.5
                speeds.append(float(b["speed_est"]))

        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0

        return {
            "success": success,
            "duration_s": duration,
            "path_length_m": path_len,
            "mean_speed_mps": mean_speed,
            "max_speed_mps": max_speed,
            "num_samples": len(self._traj_points),
            "scenario_path": self.scenario_path,
            "drone_namespace": self.drone_namespace,
            "planner": self.planner,
            "git_commit": self.git_commit,
        }

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)


class _Tee:
    """
    Tee stdout/stderr to a file without changing every print statement.
    """
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self._orig_out = sys.stdout
        self._orig_err = sys.stderr
        self._fh = None

    def start(self) -> None:
        self._fh = open(self.filepath, "w", buffering=1)
        sys.stdout = _TeeStream(self._orig_out, self._fh)
        sys.stderr = _TeeStream(self._orig_err, self._fh)

    def stop(self) -> None:
        sys.stdout = self._orig_out
        sys.stderr = self._orig_err
        if self._fh is not None:
            self._fh.close()


class _TeeStream:
    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b

    def write(self, s: str) -> int:
        self.a.write(s)
        self.b.write(s)
        return len(s)

    def flush(self) -> None:
        self.a.flush()
        self.b.flush()


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    # yaw from quaternion (Z axis rotation)
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    import math
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t0, t1)
