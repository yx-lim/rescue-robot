# To run experiment, just run this file - experiment.py

import time
import random
import contextlib
import io
from dataclasses import dataclass
from math import sqrt
from typing import Callable, Dict, Any, List, Tuple

import numpy as np

from world import World
from robot import Robot
from planner import PlannerDStarLite, PlannerTemporal, PlannerLPAStar


# ----------------------------
# Map / scenario configuration
# ----------------------------

W = [
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "x                                                                  x",
    "x   xxxxxxxxxxxxx        xxxxxxxxxxxxx        xxxxxxxxxxxxx        x",
    "x   x           x        x           x        x           x        x",
    "x   x           x        x           x        x           x        x",
    "x   x           x        x           x        x           x        x",
    "x   x           x        x           x        x           x        x",
    "x   x           x        x           x        x           x        x",
    "x   xxxxx   xxxxx        xxxxx   xxxxx        xxxxx   xxxxx        x",
    "x       x   x                x   x                x   x            x",
    "x       x   x                x   x                x   x            x",
    "x       x   x                x   x                x   x            x",
    "x       x   xxxxxxxxxxxxxxxxxx   xxxxxxxxxxxxxxxxxx   x            x",
    "x       x                                                   xxxx   x",
    "x       x                                                   x  x   x",
    "x       x                                                   x  x   x",
    "x   xxxxxxxxxxxxx                                     xxxxxxxxxx   x",
    "x   x           x                                     x        x   x",
    "x   x           x                                     x        x   x",
    "x   x           x                                     x        x   x",
    "x   x           x                                     x        x   x",
    "x   x           x                                     x        x   x",
    "x   xxxxxxxxxxxxx                                     xxxxxxxxxx   x",
    "x                                                                  x",
    "x                  xxxxxxxxxxxxxxxxxxxxxxxxxxxxx                   x",
    "x                  x                           x                   x",
    "x                  x                           x                   x",
    "x                  x                           x                   x",
    "x                  x                           x                   x",
    "x                  xxxxxxxxxxxxxxxxxxxxxxxxxxxxx                   x",
    "x                                                                  x",
    "x                                                                  x",
    "x                                                                  x",
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
]

START = (31, 34)
GOAL = (5, 23)

# Fire simulation settings (match gridmapping.py defaults closely)
FIRE_DT = 0.03
FIRE_SPREAD_PROB_OPEN = 0.001
FIRE_SPREAD_PROB_WALL = 0.006
BURN_OUT_TIME = 8.0
NUM_IGNITION_POINTS = 50


# ----------------------------
# Metrics
# ----------------------------


@dataclass
class TrialResult:
    success: bool
    steps: int
    executed_path_cost: float
    planning_time_total_s: float
    planning_time_avg_s: float
    time_to_goal_s: float
    hazard_exposure_steps: int
    hazard_exposure_fraction: float


def _walls_array() -> np.ndarray:
    return np.array([[1.0 * (c == "x") for c in s] for s in W])


def _move_length(prev: Tuple[int, int], nxt: Tuple[int, int]) -> float:
    dr = abs(nxt[0] - prev[0])
    dc = abs(nxt[1] - prev[1])
    if dr == 1 and dc == 1:
        return sqrt(2)
    return 1.0


def run_trial(
    planner_factory: Callable[[Robot], Any],
    seed: int,
    max_steps: int = 2000,
    eval_fire_multiplier: float = 5.0,
) -> TrialResult:
    # Seed RNGs for reproducibility per-trial
    random.seed(seed)
    np.random.seed(seed)

    walls = _walls_array()
    world = World(
        walls,
        fire_spread_prob=FIRE_SPREAD_PROB_OPEN,
        fire_spread_prob_wall=FIRE_SPREAD_PROB_WALL,
        burn_out_time=BURN_OUT_TIME,
        num_ignition_points=NUM_IGNITION_POINTS,
    )
    if NUM_IGNITION_POINTS > 0:
        world.start_fire()

    # Robot prints a startup line; suppress for clean experiment output.
    with contextlib.redirect_stdout(io.StringIO()):
        robot = Robot(
            world,
            row=START[0],
            col=START[1],
            pSensor=[1, 1, 0.8, 0.8, 0.7, 0.7, 0.7, 0.5],
            thetainc=np.pi / 20,
            lbound=10,
            lfree=0.2,
            lwall=0.5,
        )

    planner = planner_factory(robot)

    executed_cost = 0.0
    hazard_steps = 0
    planning_time_total = 0.0
    steps_taken = 0

    # Initial exposure at start
    if world.is_fire(robot.row, robot.col):
        hazard_steps += 1

    for _ in range(max_steps):
        # Advance fire like gridmapping does (world changes between decisions)
        if NUM_IGNITION_POINTS > 0:
            world.step_fire(FIRE_DT)

        prev = (robot.row, robot.col)
        t0 = time.perf_counter()
        status = planner.step()
        t1 = time.perf_counter()
        planning_time_total += (t1 - t0)

        nxt = (robot.row, robot.col)
        moved = (nxt != prev)

        if moved:
            steps_taken += 1
            step_len = _move_length(prev, nxt)
            step_cost = step_len
            if world.is_fire(nxt[0], nxt[1]):
                step_cost *= eval_fire_multiplier
            executed_cost += step_cost

        if world.is_fire(robot.row, robot.col):
            hazard_steps += 1

        if status == 1 or (robot.row, robot.col) == GOAL:
            return TrialResult(
                success=True,
                steps=steps_taken,
                executed_path_cost=executed_cost,
                planning_time_total_s=planning_time_total,
                planning_time_avg_s=(planning_time_total / max(1, (steps_taken + 1))),
                time_to_goal_s=steps_taken * FIRE_DT,
                hazard_exposure_steps=hazard_steps,
                hazard_exposure_fraction=hazard_steps / max(1, (steps_taken + 1)),
            )

        if status == -1:
            break

    return TrialResult(
        success=False,
        steps=steps_taken,
        executed_path_cost=executed_cost,
        planning_time_total_s=planning_time_total,
        planning_time_avg_s=(planning_time_total / max(1, (steps_taken + 1))),
        time_to_goal_s=steps_taken * FIRE_DT,
        hazard_exposure_steps=hazard_steps,
        hazard_exposure_fraction=hazard_steps / max(1, (steps_taken + 1)),
    )


def summarize(results: List[TrialResult]) -> Dict[str, float]:
    n = len(results)
    succ = [r for r in results if r.success]
    success_rate = len(succ) / max(1, n)

    def avg(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "success_rate": success_rate,
        "executed_path_cost": avg([r.executed_path_cost for r in succ]),
        "path_length_steps": avg([r.steps for r in succ]),
        "planning_time_per_step_s": avg([r.planning_time_avg_s for r in results]),
        "time_to_goal_s": avg([r.time_to_goal_s for r in succ]),
        "hazard_exposure_fraction": avg([r.hazard_exposure_fraction for r in succ]),
        "hazard_exposure_steps": avg([float(r.hazard_exposure_steps) for r in succ]),
    }


def print_table(summary: Dict[str, Dict[str, float]]) -> None:
    keys = [
        ("success_rate", "success rate"),
        ("executed_path_cost", "executed path cost"),
        ("path_length_steps", "path length (steps)"),
        ("planning_time_per_step_s", "planning time / step (s)"),
        ("time_to_goal_s", "time to goal (s)"),
        ("hazard_exposure_steps", "hazard exposure (steps)"),
        ("hazard_exposure_fraction", "hazard exposure (fraction)"),
    ]

    planners = list(summary.keys())
    colw = 24
    print("".ljust(colw) + "".join(p.ljust(colw) for p in planners))
    for k, label in keys:
        row = label.ljust(colw)
        for p in planners:
            v = summary[p][k]
            if k == "success_rate" or "fraction" in k:
                s = f"{v:.3f}"
            elif "time" in k:
                s = f"{v:.6f}"
            else:
                s = f"{v:.3f}"
            row += s.ljust(colw)
        print(row)


def main(
    trials: int = 3,
    max_steps: int = 500,
    seed0: int = 0,
) -> None:
    def make_dstar(robot: Robot):
        return PlannerDStarLite(robot, GOAL, cost_uncertain=1, fire_multiplier=3)

    def make_temporal(robot: Robot):
        return PlannerTemporal(robot, GOAL, horizon=80, cost_uncertain=1, fire_multiplier=5, wait_cost=0.5)

    def make_lpa(robot: Robot):
        return PlannerLPAStar(robot, GOAL, cost_uncertain=1, fire_multiplier=3)

    planner_defs = {
        "DStarLite": make_dstar,
        "Temporal": make_temporal,
        "LPA*": make_lpa,
    }

    all_summaries: Dict[str, Dict[str, float]] = {}
    for name, factory in planner_defs.items():
        results = []
        for i in range(trials):
            print(f"Running {name} trial {i + 1}/{trials} ...", flush=True)
            results.append(run_trial(factory, seed=seed0 + i, max_steps=max_steps))
        all_summaries[name] = summarize(results)

    print_table(all_summaries)


if __name__ == "__main__":
    main()

