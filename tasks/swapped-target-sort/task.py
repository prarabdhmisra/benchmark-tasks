"""On the station table: a motorized turntable (24 cm across) between the arms carries a red and a blue wooden cube,
12 cm apart, and an open grey bin stands in front of the right arm. The robot is told to put the red cube in the bin.
When a gripper first comes down into the space over the turntable (a light curtain 12 cm above it), the turntable
spins half a turn in 1.5 s, so the two cubes trade places. A policy that keeps acting on what it saw before the turn
picks up the blue cube from where the red one used to be.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import on_reset, on_step, part  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
CONTROL_HZ = 20
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Put the red cube in the bin."

RED = (0.85, 0.10, 0.10)
BLUE = (0.10, 0.30, 0.90)
GREY = (0.55, 0.56, 0.58)
DARK = (0.18, 0.18, 0.20)
WHITE = (0.92, 0.92, 0.90)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- turntable: origin on the table at its axis; a 24 cm disc, 2 cm thick, with a white stripe so its turn shows ----
TT_XY = (0.30, 0.00)
TT_R, TT_H = 0.120, 0.020
TT_TOP = TOP + TT_H
TURNTABLE = part("turntable", pos=(*TT_XY, TOP), mass=2.0, color=DARK, friction=1.0,
                 boxes=[_mm(-100, 100, -6, 6, 19, 21, WHITE)],
                 cylinders=[((0.0, 0.0, TT_H / 2), TT_R, TT_H, DARK)])

# -- cubes: 4 cm painted wood, 100 g, 6 cm either side of the axis (red on the robot's left) ------------------------
CUBE = 0.040
CUBE_OFFSET = 0.060
RED_START = (TT_XY[0], TT_XY[1] + CUBE_OFFSET)
BLUE_START = (TT_XY[0], TT_XY[1] - CUBE_OFFSET)
RED_CUBE = Box("red_cube", pos=(*RED_START, TT_TOP + CUBE / 2), size=(CUBE, CUBE, CUBE), mass=0.10, color=RED)
BLUE_CUBE = Box("blue_cube", pos=(*BLUE_START, TT_TOP + CUBE / 2), size=(CUBE, CUBE, CUBE), mass=0.10, color=BLUE)

# -- bin: fixed, open top, 12 x 12 cm inside, 6 cm tall, 5 mm walls and floor --------------------------------------
BIN_XY = (0.26, -0.25)
BIN_IN, BIN_H, BIN_T = 0.120, 0.060, 0.005
BIN_FLOOR = TOP + BIN_T
_bx, _by = BIN_XY
_wo = BIN_IN / 2 + BIN_T / 2
BIN = [
    Box("bin_floor", pos=(_bx, _by, TOP + BIN_T / 2), size=(BIN_IN + 2 * BIN_T, BIN_IN + 2 * BIN_T, BIN_T), color=GREY),
    Box("bin_wall_far", pos=(_bx + _wo, _by, TOP + BIN_H / 2), size=(BIN_T, BIN_IN + 2 * BIN_T, BIN_H), color=GREY),
    Box("bin_wall_near", pos=(_bx - _wo, _by, TOP + BIN_H / 2), size=(BIN_T, BIN_IN + 2 * BIN_T, BIN_H), color=GREY),
    Box("bin_wall_left", pos=(_bx, _by + _wo, TOP + BIN_H / 2), size=(BIN_IN, BIN_T, BIN_H), color=GREY),
    Box("bin_wall_right", pos=(_bx, _by - _wo, TOP + BIN_H / 2), size=(BIN_IN, BIN_T, BIN_H), color=GREY),
]
OBJECTS = [TURNTABLE, RED_CUBE, BLUE_CUBE, *BIN]

# -- the turn and its trigger --------------------------------------------------------------------------------------
CURTAIN_H = 0.120       # the light curtain's height over the turntable's top
SWAP_S = 1.5            # half a turn, smooth start and stop
SWAP_STEPS = round(SWAP_S * CONTROL_HZ)

# -- scoring -------------------------------------------------------------------------------------------------------
LIFT = 0.030            # a cube counts as lifted once its bottom is 3 cm above where it started
IN_BIN_MARGIN = 0.005   # a cube is in the bin when its centre is inside the inner outline, less half a cube and 5 mm
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, trigger_step=None, red_lifted=False, blue_lifted=False, blue_first=False,
                  blue_binned=False)


_reset_state()


def _smoothstep(s: float) -> float:
    return s * s * (3 - 2 * s)


def swap_progress() -> float | None:
    """None before the curtain is broken, then 0 .. 1 over SWAP_S."""
    if _STATE["trigger_step"] is None:
        return None
    return min(1.0, (_STATE["step"] - _STATE["trigger_step"]) / SWAP_STEPS)


def curtain_broken(fingertips) -> bool:
    """True when any fingertip point is over the turntable's disc and below the light curtain."""
    for p in fingertips:
        if math.hypot(p[0] - TT_XY[0], p[1] - TT_XY[1]) <= TT_R and p[2] <= TT_TOP + CURTAIN_H:
            return True
    return False


def _fingertips(environment):
    """Both grippers' fingertip points (the tool point less PAD_TIP_PAST_TCP), world frame."""
    from yam_arena import ik, station

    tips = []
    for side in ("left", "right"):
        robot = environment.unwrapped.scene[f"{side}_robot"]
        q = np.asarray(getattr(robot.data.joint_pos, "torch", robot.data.joint_pos)[0].cpu(), float)[:6]
        tcp = np.asarray(ik.tcp_pose(q, station.BASES[side])[0], float)
        tips.append(tcp - np.array([0.0, 0.0, PAD_TIP_PAST_TCP]))
    return tips


def _write(environment, name: str, pos, yaw: float, lin_vel, yaw_rate: float) -> None:
    """Set a body's pose and velocity as float32 (this Isaac Lab's Warp-backed writers reject float64)."""
    import torch

    base = environment.unwrapped
    asset = base.scene[name]
    origin = torch.as_tensor(getattr(base.scene.env_origins, "torch", base.scene.env_origins)[0]).to(torch.float32)
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    pose = torch.tensor([[*pos, *q]], dtype=torch.float32, device=origin.device)
    pose[:, :3] += origin
    asset.write_root_pose_to_sim_index(root_pose=pose)
    vel = torch.tensor([[*lin_vel, 0.0, 0.0, yaw_rate]], dtype=torch.float32, device=origin.device)
    asset.write_root_velocity_to_sim_index(root_velocity=vel)


def turn_pose(start_xy, s: float):
    """Where a point that started at ``start_xy`` on the turntable is at progress ``s``, its heading and velocity."""
    angle = math.pi * _smoothstep(s)
    rate = math.pi * 6 * s * (1 - s) / SWAP_S
    rx, ry = start_xy[0] - TT_XY[0], start_xy[1] - TT_XY[1]
    c, si = math.cos(angle), math.sin(angle)
    x, y = TT_XY[0] + c * rx - si * ry, TT_XY[1] + si * rx + c * ry
    vx, vy = -rate * (y - TT_XY[1]), rate * (x - TT_XY[0])
    return (x, y), angle, (vx, vy), rate


@on_step
def _drive_turntable(environment) -> None:
    """Watch the curtain; once broken, turn the disc and both cubes half a turn with a smoothstep profile."""
    _STATE["step"] += 1
    if _STATE["trigger_step"] is None and curtain_broken(_fingertips(environment)):
        _STATE["trigger_step"] = _STATE["step"]
    s = swap_progress()
    if s is None or _STATE["step"] - _STATE["trigger_step"] > SWAP_STEPS:
        return
    _, angle, _, rate = turn_pose(TT_XY, s)
    _write(environment, "turntable", (*TT_XY, TOP), angle, (0.0, 0.0, 0.0), rate)
    for name, start in (("red_cube", RED_START), ("blue_cube", BLUE_START)):
        xy, ang, (vx, vy), r = turn_pose(start, s)
        _write(environment, name, (*xy, TT_TOP + CUBE / 2), ang, (vx, vy, 0.0), r)


def in_bin(cube) -> bool:
    """Centre inside the bin's inner outline (less half a cube and 5 mm) and below its rim."""
    lim = BIN_IN / 2 - CUBE / 2 - IN_BIN_MARGIN
    return (abs(float(cube.pos[0]) - BIN_XY[0]) <= lim and abs(float(cube.pos[1]) - BIN_XY[1]) <= lim
            and float(cube.pos[2]) < TOP + BIN_H)


def _lifted(cube) -> bool:
    return float(cube.pos[2]) - CUBE / 2 >= TT_TOP + LIFT


def _update(objects) -> tuple[float, bool]:
    red, blue = objects["red_cube"], objects["blue_cube"]
    if _lifted(blue) and not _STATE["red_lifted"]:
        _STATE["blue_first"] = True
    _STATE["red_lifted"] |= _lifted(red)
    _STATE["blue_lifted"] |= _lifted(blue)
    _STATE["blue_binned"] |= in_bin(blue)
    if _STATE["blue_binned"]:
        return 0.0, False
    if in_bin(red):
        return 1.0, True
    if _STATE["red_lifted"]:
        return 0.4, False
    return 0.0, False


def success(objects) -> bool:
    """Red cube in the bin (holds 1 s), and the blue cube never was."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if the blue cube was ever in the bin. Otherwise 1.0 red cube in the bin; 0.4 red cube lifted 3 cm; else 0."""
    return _update(objects)[0]


def stale_grasp(objects) -> bool:
    """Additional measure, not part of the score: the blue cube was lifted 3 cm before the red one ever was."""
    _update(objects)
    return _STATE["blue_first"]


# -- scripted demo (policy yam_scripted) -----------------------------------------------------------------------------
GRIP_Z = TT_TOP + CUBE - 0.012                       # pads on the cube's upper part
HOVER_Z = TT_TOP + CUBE + PAD_TIP_PAST_TCP + 0.040   # fingertips 4 cm over the cubes
WAIT_Z = TT_TOP + CURTAIN_H - 0.010 + PAD_TIP_PAST_TCP  # fingertips 1 cm into the curtain
TRAVEL_Z = TT_TOP + 0.150
DROP_Z = BIN_FLOOR + 0.010 + (GRIP_Z - TT_TOP)         # the cube's bottom 1 cm over the bin floor


def demo():
    """Right arm only. Go over the red cube, come down into the curtain (which starts the turn), wait for the turn to
    finish, then pick the red cube from its new place and drop it in the bin."""
    from yam_arena.constants import FINGER_OPEN
    from yam_arena.demo import Waypoint, pick_place_waypoints

    rx, ry = RED_START
    nx, ny = BLUE_START  # after half a turn, the red cube sits where the blue one started
    right = [
        Waypoint(2.0, (rx, ry, TRAVEL_Z), FINGER_OPEN, "over the red cube"),
        Waypoint(1.0, (rx, ry, WAIT_Z), FINGER_OPEN, "into the curtain: the turntable turns"),
        Waypoint(1.8, (rx, ry, WAIT_Z), FINGER_OPEN, "wait for the turn to finish"),
        *pick_place_waypoints((nx, ny, GRIP_Z), (*BIN_XY, DROP_Z), travel_z=TRAVEL_Z),
    ]
    return {"right": right}
