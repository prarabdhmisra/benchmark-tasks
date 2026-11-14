"""Checks of task.py's scoring, trigger and demo, without the simulator. Needs yam_arena (the official stack):

    python -m pytest -q test_task.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).with_name("task.py"))
M = TASK.module
REST_Z = M.TT_TOP + M.CUBE / 2


def cube(x, y, z=REST_Z):
    c = np.array([x, y, z])
    h = M.CUBE / 2
    return ObjectState(pos=c, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                       low=c - h, high=c + h)


def scene(red, blue):
    return {"red_cube": red, "blue_cube": blue}


BIN_Z = M.BIN_FLOOR + M.CUBE / 2
SWAPPED = dict(red=cube(*M.BLUE_START), blue=cube(*M.RED_START))


@pytest.fixture(autouse=True)
def fresh():
    M._reset_state()


def test_the_start_scores_zero():
    assert M.score(scene(cube(*M.RED_START), cube(*M.BLUE_START))) == 0.0


def test_red_in_the_bin_is_success():
    s = scene(cube(*M.BIN_XY, BIN_Z), SWAPPED["blue"])
    assert M.success(s) and M.score(s) == 1.0


def test_red_lifted_scores_0_4():
    assert M.score(scene(cube(*M.BLUE_START, REST_Z + 0.031), SWAPPED["blue"])) == 0.4
    M._reset_state()
    assert M.score(scene(cube(*M.BLUE_START, REST_Z + 0.029), SWAPPED["blue"])) == 0.0


def test_blue_in_the_bin_scores_zero_for_the_rest_of_the_trial():
    assert M.score(scene(SWAPPED["red"], cube(*M.BIN_XY, BIN_Z))) == 0.0
    assert M.score(scene(cube(*M.BIN_XY, BIN_Z), cube(0.40, 0.20))) == 0.0


def test_both_in_the_bin_scores_zero():
    assert M.score(scene(cube(M.BIN_XY[0] - 0.03, M.BIN_XY[1], BIN_Z), cube(M.BIN_XY[0] + 0.03, M.BIN_XY[1], BIN_Z))) == 0.0


@pytest.mark.parametrize(("dx", "inside"), [(0.034, True), (0.036, False)])
def test_the_bin_outline(dx, inside):
    assert M.score(scene(cube(M.BIN_XY[0] + dx, M.BIN_XY[1], BIN_Z), SWAPPED["blue"])) == (1.0 if inside else 0.0)


def test_red_held_over_the_bin_is_lifted_but_not_in():
    assert M.score(scene(cube(*M.BIN_XY, M.TOP + M.BIN_H + 0.02), SWAPPED["blue"])) == 0.4


def test_lifting_blue_first_is_a_stale_grasp():
    M.score(scene(SWAPPED["red"], cube(*M.RED_START, REST_Z + 0.05)))
    assert M.stale_grasp(scene(SWAPPED["red"], SWAPPED["blue"]))


def test_lifting_red_first_is_not_a_stale_grasp():
    M.score(scene(cube(*M.BLUE_START, REST_Z + 0.05), SWAPPED["blue"]))
    assert not M.stale_grasp(scene(cube(*M.BLUE_START, REST_Z + 0.05), cube(*M.RED_START, REST_Z + 0.05)))


def test_the_turn_swaps_the_cubes():
    (x, y), angle, _, rate = M.turn_pose(M.RED_START, 1.0)
    assert (x, y) == pytest.approx(M.BLUE_START, abs=1e-9) and angle == pytest.approx(math.pi) and rate == 0.0
    (x, y), _, (vx, vy), _ = M.turn_pose(M.RED_START, 0.5)
    assert math.hypot(x - M.TT_XY[0], y - M.TT_XY[1]) == pytest.approx(M.CUBE_OFFSET)
    assert math.hypot(vx, vy) == pytest.approx(math.pi * 1.5 / M.SWAP_S * M.CUBE_OFFSET)


@pytest.mark.parametrize(("tip", "broken"), [((0.30, 0.06, None), True), ((0.30, 0.13, None), False),
                                              ((0.30, 0.06, 0.005), False)])
def test_the_curtain(tip, broken):
    z = M.TT_TOP + M.CURTAIN_H - 0.001 if tip[2] is None else M.TT_TOP + M.CURTAIN_H + tip[2]
    assert M.curtain_broken([np.array([tip[0], tip[1], z])]) is broken


def test_the_arms_at_rest_do_not_break_the_curtain():
    tips = [np.asarray(ik.tcp_pose(np.array(READY, float), station.BASES[s])[0]) - [0, 0, PAD_TIP_PAST_TCP]
            for s in ("left", "right")]
    assert not M.curtain_broken(tips), tips


def test_the_cubes_fit_through_the_turn_without_touching():
    assert 2 * M.CUBE_OFFSET > M.CUBE * math.sqrt(2) and M.CUBE_OFFSET + M.CUBE / math.sqrt(2) < M.TT_R


def test_the_demo_waits_inside_the_curtain_and_clear_of_the_cubes():
    assert M.WAIT_Z - PAD_TIP_PAST_TCP < M.TT_TOP + M.CURTAIN_H
    assert M.WAIT_Z - PAD_TIP_PAST_TCP > M.TT_TOP + M.CUBE + 0.03


def test_the_demo_is_within_reach():
    plan = ArmPlan(M.demo()["right"], station.BASES["right"])
    for t in np.arange(0.0, plan.duration, 0.05):
        plan.action7(t)
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)
