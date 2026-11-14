"""Checks of task.py's scoring and demo, without the simulator. Needs yam_arena (the official stack):

    python -m pytest -q test_task.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("task.py"))
M = TASK.module
ON_PAD = M.TOP + M.PAD_H


def pose(name, x, y, z, tilt_deg=0.0):
    """A part's state with its origin at x, y, z, tilted ``tilt_deg`` about world x."""
    h = math.radians(tilt_deg) / 2
    return M.PartState(pos=np.array([x, y, z]), quat_xyzw=np.array([math.sin(h), 0.0, 0.0, math.cos(h)]),
                       lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.zeros(3), high=np.zeros(3))


def tub_at(xy, tilt_deg=0.0):
    return pose("tub", *xy, M.TOP, tilt_deg)


def lid_seated(xy, dx=0.0, dz=0.0, tilt_deg=0.0):
    return pose("lid", xy[0] + dx, xy[1], M.TOP + M.RIM_Z + dz, tilt_deg)


def scene(tub, lid):
    return {"tub": tub, "lid": lid}


@pytest.fixture(autouse=True)
def fresh():
    M._reset_state()


def lifted_and_moved():
    """The lid has left its pad and the stage has finished its move."""
    M.score(scene(tub_at(M.TUB_START), pose("lid", *M.LID_XY, ON_PAD + 0.05)))
    M._STATE["step"] += M.SLIDE_STEPS


def test_the_start_scores_zero_and_the_stage_waits():
    start = scene(tub_at(M.TUB_START), pose("lid", *M.LID_XY, ON_PAD))
    assert M.score(start) == 0.0 and not M.success(start)
    assert M.slide_progress() is None


def test_lifting_the_lid_starts_the_stage_and_scores_0_2():
    assert M.score(scene(tub_at(M.TUB_START), pose("lid", *M.LID_XY, ON_PAD + 0.011))) == 0.2
    assert M.slide_progress() == 0.0


def test_a_lid_seated_on_the_moved_tub_is_success():
    lifted_and_moved()
    s = scene(tub_at(M.TUB_END), lid_seated(M.TUB_END))
    assert M.success(s) and M.score(s) == 1.0


@pytest.mark.parametrize(("dx", "expected"), [(0.014, 1.0), (0.016, 0.5), (0.049, 0.5), (0.051, 0.2)])
def test_offset_bands(dx, expected):
    lifted_and_moved()
    assert M.score(scene(tub_at(M.TUB_END), lid_seated(M.TUB_END, dx=dx))) == expected


def test_a_tilted_lid_is_on_the_tub_but_not_seated():
    lifted_and_moved()
    assert M.score(scene(tub_at(M.TUB_END), lid_seated(M.TUB_END, dz=0.004, tilt_deg=8))) == 0.5


def test_the_lid_lowered_where_the_tub_used_to_be_is_stale():
    lifted_and_moved()
    s = scene(tub_at(M.TUB_END), pose("lid", *M.TUB_START, M.TOP + M.BED_H))
    assert M.score(s) == 0.2 and M.stale_placement(s)


def test_a_knocked_tub_scores_zero_for_the_rest_of_the_trial():
    lifted_and_moved()
    assert M.score(scene(tub_at(M.TUB_END, tilt_deg=15), lid_seated(M.TUB_END))) == 0.0
    assert M.score(scene(tub_at(M.TUB_END), lid_seated(M.TUB_END))) == 0.0


def test_pushing_the_tub_before_the_lift_scores_zero():
    assert M.score(scene(tub_at(M.TUB_START + [0.0, 0.03]), pose("lid", *M.LID_XY, ON_PAD))) == 0.0


def test_the_stage_moving_the_tub_is_not_a_knock():
    M.score(scene(tub_at(M.TUB_START), pose("lid", *M.LID_XY, ON_PAD + 0.05)))
    M._STATE["step"] += M.SLIDE_STEPS // 2
    assert M.score(scene(tub_at(M.TUB_START + M.SLIDE / 2), pose("lid", *M.LID_XY, ON_PAD + 0.05))) == 0.2


def test_seating_before_the_move_finishes_is_not_success():
    M.score(scene(tub_at(M.TUB_START), pose("lid", *M.LID_XY, ON_PAD + 0.05)))
    assert not M.success(scene(tub_at(M.TUB_START), lid_seated(M.TUB_START)))


def test_a_seated_lid_covers_the_whole_opening():
    inner_half_diagonal = math.hypot(0.046, 0.046)
    assert M.SEAT_OFFSET + inner_half_diagonal < M.LID_R


def test_the_demo_grips_the_knob_above_the_lid():
    assert M.GRIP_Z - M.PAD_TIP_PAST_TCP > M.LID_TOP + 0.005  # fingertips clear the lid's top
    assert M.GRIP_Z < M.LID_TOP + M.KNOB_H - 0.010             # tool point at least 1 cm into the knob


def test_the_demo_is_within_reach():
    plan = ArmPlan(M.demo()["right"], station.BASES["right"])
    for t in np.arange(0.0, plan.duration, 0.05):
        plan.action7(t)
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)
