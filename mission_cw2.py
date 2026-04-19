#\!/usr/bin/env python3
"""mission_cw2.py – CW2 Multi-drone Swarm Mission.

Top-level entrypoint for the CW2 challenge.  Loads a scenario YAML,
creates a conductor (centralised or decentralised Boids), arms and takes
off all drones, then executes each stage in sequence.

Usage (from the challenge_multi_drone directory, after sourcing the
workspace with ``source setup.bash``):

    python3 mission_cw2.py [options]

Options
-------
--scenario PATH      Scenario YAML file (default: scenarios/scenario1.yaml)
--namespaces NS...   Drone namespaces (default: drone0..drone2)
--stages STAGES      Comma-separated stage numbers to run (default: 1,2,3,4)
--approach MODE      centralised (default) or decentralised (Boids)
--speed SPEED        Override default cruise speed for all stages (m/s)
--takeoff-height H   Takeoff height in metres (default: 1.2)
--no-sim-time        Disable use_sim_time (for real hardware)
--verbose            Print stage progress messages

Both --approach centralised and --approach decentralised satisfy the CW2
requirement for demonstrating both control paradigms.
"""

__authors__ = "CW2 student implementation"

import argparse
import sys
import os
import time
import yaml
import rclpy

from swarm.drone_agent import DroneAgent
from swarm.swarm_conductor import SwarmConductor
from swarm.boids_conductor import BoidsConductor
from stages.stage1_formation import run_stage1
from stages.stage2_windows import run_stage2
from stages.stage3_forest import run_stage3
from stages.stage4_dynamic import run_stage4


# --------------------------------------------------------------------------- #
#  CLI argument parsing                                                        #
# --------------------------------------------------------------------------- #

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CW2 multi-drone swarm mission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        default="scenarios/scenario1.yaml",
        help="Path to scenario YAML file",
    )
    parser.add_argument(
        "--namespaces",
        nargs="+",
        default=["drone0", "drone1", "drone2"],
        metavar="NS",
        help="ROS2 drone namespaces",
    )
    parser.add_argument(
        "--stages",
        default="1,2,3,4",
        help="Comma-separated stage numbers to run (e.g. 1,3)",
    )
    parser.add_argument(
        "--approach",
        choices=["centralised", "decentralised"],
        default="centralised",
        help=(
            "centralised: SwarmConductor (global formation commands); "
            "decentralised: BoidsConductor (local flocking rules)."
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Override default cruise speed for all stages (m/s)",
    )
    parser.add_argument(
        "--takeoff-height",
        type=float,
        default=1.2,
        dest="takeoff_height",
        help="Takeoff height (m)",
    )
    parser.add_argument(
        "--no-sim-time",
        action="store_true",
        dest="no_sim_time",
        help="Disable use_sim_time (for real hardware)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stage progress messages",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
#  Scenario loading                                                            #
# --------------------------------------------------------------------------- #

def _load_scenario(path: str) -> dict:
    """Load the scenario dictionary from a YAML file.

    Searches relative to the current working directory first, then
    relative to the directory containing this script.
    """
    candidates = [
        path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            with open(candidate) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        "Scenario file not found: " + repr(path) + "\n"
        "Searched: " + repr(candidates)
    )


# --------------------------------------------------------------------------- #
#  Stage dispatch                                                              #
# --------------------------------------------------------------------------- #

def _run_stages(conductor, scenario: dict, stage_numbers: list,
                speed_override, verbose: bool) -> None:
    """Execute each requested stage in order."""
    stage_size = tuple(scenario.get("stage_size", [10.0, 10.0]))

    for stage_num in stage_numbers:

        if stage_num == 1:
            if "stage1" not in scenario:
                print("[Mission] Warning: stage1 not in scenario, skipping.")
                continue
            print("\n" + "=" * 60)
            print("STAGE 1: Formation flying (circle trajectory)")
            print("=" * 60)
            kwargs = {}
            if speed_override:
                kwargs["speed"] = speed_override
            run_stage1(conductor, scenario["stage1"], verbose=verbose, **kwargs)

        elif stage_num == 2:
            if "stage2" not in scenario:
                print("[Mission] Warning: stage2 not in scenario, skipping.")
                continue
            print("\n" + "=" * 60)
            print("STAGE 2: Window traversal (columnN formation)")
            print("=" * 60)
            kwargs = {}
            if speed_override:
                kwargs["speed"] = speed_override
            run_stage2(conductor, scenario["stage2"], verbose=verbose, **kwargs)

        elif stage_num == 3:
            if "stage3" not in scenario:
                print("[Mission] Warning: stage3 not in scenario, skipping.")
                continue
            print("\n" + "=" * 60)
            print("STAGE 3: Forest navigation (A* path planning)")
            print("=" * 60)
            kwargs = {}
            if speed_override:
                kwargs["speed"] = speed_override
            run_stage3(conductor, scenario["stage3"], verbose=verbose, **kwargs)

        elif stage_num == 4:
            if "stage4" not in scenario:
                print("[Mission] Warning: stage4 not in scenario, skipping.")
                continue
            print("\n" + "=" * 60)
            print("STAGE 4: Dynamic obstacles (RRT* + online replanning)")
            print("=" * 60)
            kwargs = dict(stage_size=stage_size)
            if speed_override:
                kwargs["speed"] = speed_override
            run_stage4(conductor, scenario["stage4"], verbose=verbose, **kwargs)

        else:
            print("[Mission] Unknown stage number " + str(stage_num) + ", skipping.")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    args = _parse_args(argv)

    # Parse stage list
    try:
        stage_numbers = [int(s.strip()) for s in args.stages.split(",")
                         if s.strip()]
    except ValueError:
        print("ERROR: Invalid --stages value: " + repr(args.stages))
        return 1

    use_sim_time = not args.no_sim_time

    # Load scenario
    try:
        scenario = _load_scenario(args.scenario)
    except FileNotFoundError as exc:
        print("ERROR: " + str(exc))
        return 1

    print("[Mission] Scenario : " + str(scenario.get("name", args.scenario)))
    print("[Mission] Stages   : " + str(stage_numbers))
    print("[Mission] Approach : " + args.approach)
    print("[Mission] Drones   : " + str(args.namespaces))

    # Initialise rclpy
    rclpy.init()

    # Select conductor — each conductor type creates its own DroneAgent objects
    # so there is exactly one ROS2 node per drone namespace at all times.
    if args.approach == "decentralised":
        print("[Mission] Using BoidsConductor (decentralised Boids flocking).")
        drones = [
            DroneAgent(ns, verbose=args.verbose, use_sim_time=use_sim_time)
            for ns in args.namespaces
        ]
        conductor = BoidsConductor(drones, verbose=args.verbose)
    else:
        print("[Mission] Using SwarmConductor (centralised formation commands).")
        conductor = SwarmConductor(
            namespaces=args.namespaces,
            verbose=args.verbose,
            use_sim_time=use_sim_time,
        )

    try:
        # Wait for pose data
        print("[Mission] Waiting for pose data from all drones (may take up to 90 s)...")
        pose_ok = False
        for _attempt in range(3):
            if conductor.wait_for_poses(timeout=90.0):
                pose_ok = True
                break
            print(f"[Mission] Pose attempt {_attempt+1}/3 timed out, retrying...")
        if not pose_ok:
            print("[Mission] WARNING: Could not confirm pose data after 270 s. Proceeding.")

        # Arm and offboard
        print("[Mission] Arming and switching to offboard mode...")
        if not conductor.arm_and_offboard():
            print("[Mission] WARN: arm/offboard returned failure (likely already armed from previous run); continuing.")
            pass  # tolerate re-run; takeoff behaviour re-arms as needed

        # Takeoff
        print("[Mission] Taking off to " + str(args.takeoff_height) + " m...")
        if not conductor.takeoff(height=args.takeoff_height, timeout=45.0):
            print("[Mission] WARNING: Takeoff timeout.")

        print("[Mission] All drones airborne.  Beginning mission...")

        # Run stages
        _run_stages(
            conductor=conductor,
            scenario=scenario,
            stage_numbers=stage_numbers,
            speed_override=args.speed,
            verbose=args.verbose,
        )

        print("\n[Mission] All stages complete.  Landing...")
        conductor.land(timeout=45.0)
        print("[Mission] Landed.  Mission complete.")

    except KeyboardInterrupt:
        print("\n[Mission] Interrupted.  Landing...")
        try:
            conductor.land(timeout=20.0)
        except Exception:
            pass

    except Exception as exc:
        print("\n[Mission] Unhandled exception: " + str(exc))
        import traceback
        traceback.print_exc()
        try:
            conductor.land(timeout=20.0)
        except Exception:
            pass
        return 1

    finally:
        print("[Mission] Shutting down ROS2...")
        conductor.shutdown()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
