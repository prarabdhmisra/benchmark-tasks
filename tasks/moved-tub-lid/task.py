"""On the station table, in front of the right arm: an open white tub (10 x 10 x 7 cm) fixed to the carriage of a
motorized linear stage, and a round blue lid (17 cm across) with an upright knob, lying flat on a switch pad to the
tub's right. The robot must put the lid on the tub. Lifting the lid off its pad starts the stage: in 1 s, the tub
slides 8 cm toward the robot's left and stops there. A policy that keeps acting on what it saw before the move lowers
the lid where the tub used to be.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, on_reset, on_step, part  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
CONTROL_HZ = 20
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Put the blue lid on the white tub."

WHITE = (0.92, 0.92, 0.90)
GREY = (0.45, 0.46, 0.48)
DARK = (0.18, 0.18, 0.20)
BLUE = (0.10, 0.30, 0.90)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the tub on its carriage: origin on the table under the tub's centre; the stage's bed is 10 mm tall -----------
BED_H = 0.010
RIM_Z = 0.090   # the tub's rim above the table
TUB_BOXES = [
    _mm(-70, 70, -70, 70, 10, 20, GREY),                                          # carriage plate, 14 x 14 cm
    _mm(-50, 50, -50, 50, 20, 24, WHITE),                                         # tub floor
    _mm(-50, 50, 46, 50, 24, 90, WHITE), _mm(-50, 50, -50, -46, 24, 90, WHITE),   # walls, 4 mm thick
    _mm(46, 50, -46, 46, 24, 90, WHITE), _mm(-50, -46, -46, 46, 24, 90, WHITE),
]

# -- the lid: origin at the centre of its underside; a 6 mm disc with a 25 mm knob standing 45 mm up from it -------
LID_R, LID_T = 0.085, 0.006
KNOB_R, KNOB_H = 0.0125, 0.045
LID_CYLS = [((0.0, 0.0, LID_T / 2), LID_R, LID_T, BLUE),
            ((0.0, 0.0, LID_T + KNOB_H / 2), KNOB_R, KNOB_H, BLUE)]

# -- fixed layout (world frame: x forward from the arms, y left, metres) -------------------------------------------
TUB_START = np.array([0.36, -0.14])
SLIDE = np.array([0.0, 0.08])        # 8 cm toward the robot's left
TUB_END = TUB_START + SLIDE
SLIDE_S = 1.0                        # smooth start and stop; peak speed 12 cm/s
LID_XY = (0.24, -0.34)
PAD_H = 0.003

BED = Box("stage_bed", pos=(0.36, -0.11, TOP + BED_H / 2), size=(0.12, 0.26, BED_H), color=DARK)  # fixed
PAD = Box("switch_pad", pos=(*LID_XY, TOP + PAD_H / 2), size=(0.18, 0.18, PAD_H), color=DARK)      # fixed
TUB = part("tub", pos=(*TUB_START, TOP), mass=1.5, boxes=TUB_BOXES, color=WHITE, friction=0.6)
# part() needs at least one box: a 10 x 10 cm square inside the disc (half-diagonal 7.1 cm < 8.5 cm), same thickness.
LID = part("lid", pos=(*LID_XY, TOP + PAD_H), mass=0.080,
           boxes=[((0.0, 0.0, LID_T / 2), (0.10, 0.10, LID_T), BLUE)], cylinders=LID_CYLS, color=BLUE, friction=0.8)
OBJECTS = [BED, PAD, TUB, LID]

# -- scoring -------------------------------------------------------------------------------------------------------
LIFT_TRIGGER = 0.010    # the lid's underside 1 cm above its pad starts the stage
SEAT_OFFSET = 0.015     # seated: lid centre within 1.5 cm of the tub's centre, seen from above
SEAT_DZ = (-0.003, 0.005)
SEAT_TILT_DEG = 5.0
ON_TUB_OFFSET = 0.050   # resting on the tub, off centre
ON_TUB_DZ = (-0.010, 0.020)
TUB_TILT_DEG = 10.0
TUB_MOVE_LIMIT = 0.020
STALE_RADIUS = 0.060
SLIDE_STEPS = round(SLIDE_S * CONTROL_HZ)
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, trigger_step=None, knocked=False)


_reset_state()


def _tilt_deg(p: PartState) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(p.rotation()[2, 2])))))


def _smoothstep(s: float) -> float:
    return s * s * (3 - 2 * s)


def slide_progress() -> float | None:
    """None before the stage starts, then 0 .. 1 over SLIDE_S."""
    if _STATE["trigger_step"] is None:
        return None
    return min(1.0, (_STATE["step"] - _STATE["trigger_step"]) / SLIDE_STEPS)


def _note_lift(lid: PartState) -> None:
    if _STATE["trigger_step"] is None and float(lid.pos[2]) - (TOP + PAD_H) >= LIFT_TRIGGER:
        _STATE["trigger_step"] = _STATE["step"]


@on_step
def _drive_stage(environment) -> None:
    """The stage: once the lid has left its pad, move the tub along SLIDE with a smoothstep profile, then stop."""
    from yam_arena.scene import object_states

    _STATE["step"] += 1
    _note_lift(object_states(environment, ["lid"])["lid"])
    s = slide_progress()
    if s is None or _STATE["step"] - _STATE["trigger_step"] > SLIDE_STEPS:
        return
    speed = 6 * s * (1 - s) / SLIDE_S  # d(smoothstep)/dt
    _write_tub(environment, (*(TUB_START + _smoothstep(s) * SLIDE), TOP), (*(SLIDE * speed), 0.0))


def _write_tub(environment, pos, lin_vel) -> None:
    """Set the tub's body pose (upright) and velocity, as float32: partkit.write_pose keeps the env origin's dtype,
    and this Isaac Lab's Warp-backed writers reject float64."""
    import torch

    base = environment.unwrapped
    asset = base.scene["tub"]
    origin = getattr(base.scene.env_origins, "torch", base.scene.env_origins)[0]
    origin = torch.as_tensor(origin).to(torch.float32)
    pose = torch.tensor([[*pos, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=origin.device)
    pose[:, :3] += origin
    asset.write_root_pose_to_sim_index(root_pose=pose)
    vel = torch.tensor([[*lin_vel, 0.0, 0.0, 0.0]], dtype=torch.float32, device=origin.device)
    asset.write_root_velocity_to_sim_index(root_velocity=vel)


def lid_on_tub(tub: PartState, lid: PartState):
    """Lid centre's horizontal offset from the tub's centre (m), its underside's height above the rim (m), its tilt."""
    c = tub.to_local(lid.pos)
    return math.hypot(c[0], c[1]), float(c[2] - RIM_Z), _tilt_deg(lid)


def _update(objects) -> tuple[float, bool]:
    tub, lid = objects["tub"], objects["lid"]
    _note_lift(lid)
    s = slide_progress()
    if s is None or s >= 1.0:  # the tub is not being moved by the stage: it must stay where the stage left it
        ref = TUB_START if s is None else TUB_END
        _STATE["knocked"] |= float(np.linalg.norm(tub.pos[:2] - ref)) > TUB_MOVE_LIMIT
    _STATE["knocked"] |= _tilt_deg(tub) > TUB_TILT_DEG
    if _STATE["knocked"]:
        return 0.0, False
    if s is None:
        return 0.0, False
    off, dz, tilt = lid_on_tub(tub, lid)
    if s >= 1.0 and off <= SEAT_OFFSET and SEAT_DZ[0] <= dz <= SEAT_DZ[1] and tilt < SEAT_TILT_DEG:
        return 1.0, True
    if off <= ON_TUB_OFFSET and ON_TUB_DZ[0] <= dz <= ON_TUB_DZ[1]:
        return 0.5, False
    return 0.2, False


def success(objects) -> bool:
    """Lid seated on the tub at its new position (holds for 1 s), tub never knocked."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if the tub was knocked (tilted over 10 degrees, or pushed 2 cm from where the stage left it). Otherwise 1.0
    lid seated on the moved tub; 0.5 lid resting on the tub off centre; 0.2 lid lifted off its pad; else 0."""
    return _update(objects)[0]


def stale_placement(objects) -> bool:
    """Additional measure, not part of the score: the lid ends on the stage bed or table (underside below the tub's
    floor) with its centre within 6 cm of where the tub stood before the move."""
    lid = objects["lid"]
    return (_STATE["trigger_step"] is not None and float(lid.pos[2]) < TOP + 0.024
            and float(np.linalg.norm(lid.pos[:2] - TUB_START)) <= STALE_RADIUS)


# -- scripted demo (policy yam_scripted) -----------------------------------------------------------------------------
LID_TOP = TOP + PAD_H + LID_T
GRIP_Z = LID_TOP + PAD_TIP_PAST_TCP + 0.008       # fingertips 8 mm above the lid's top, pads on the knob
HOVER_Z = LID_TOP + KNOB_H + PAD_TIP_PAST_TCP + 0.030
GRIP_ABOVE_UNDERSIDE = GRIP_Z - (TOP + PAD_H)
CARRY_Z = TOP + RIM_Z + 0.030 + GRIP_ABOVE_UNDERSIDE   # lid's underside 3 cm over the rim
PLACE_Z = TOP + RIM_Z + 0.004 + GRIP_ABOVE_UNDERSIDE   # let go 4 mm over the rim


def demo():
    """Right arm only. Grip the knob from above, lift (which starts the stage), wait at carry height until the tub has
    stopped, carry the lid over the tub's new centre, lower it to 4 mm over the rim, let go, rise and move back."""
    lx, ly = LID_XY
    tx, ty = TUB_END
    right = [
        Waypoint(2.0, (lx, ly, HOVER_Z), FINGER_OPEN, "above the knob"),
        Waypoint(1.5, (lx, ly, GRIP_Z), FINGER_OPEN, "down to the knob"),
        Waypoint(0.4, (lx, ly, GRIP_Z), FINGER_OPEN, "settle"),
        Waypoint(0.8, (lx, ly, GRIP_Z), FINGER_CLOSED, "grip the knob"),
        Waypoint(1.5, (lx, ly, CARRY_Z), FINGER_CLOSED, "lift: the stage starts"),
        Waypoint(1.0, (lx, ly, CARRY_Z), FINGER_CLOSED, "wait for the tub to stop"),
        Waypoint(2.0, (tx, ty, CARRY_Z), FINGER_CLOSED, "over the tub's new centre"),
        Waypoint(1.5, (tx, ty, PLACE_Z), FINGER_CLOSED, "lower onto the rim"),
        Waypoint(0.6, (tx, ty, PLACE_Z), FINGER_OPEN, "let go"),
        Waypoint(1.0, (tx, ty, PLACE_Z + 0.03), FINGER_OPEN, "rise off the knob"),
        Waypoint(1.5, (lx + 0.02, ty - 0.16, CARRY_Z), FINGER_OPEN, "back clear of the tub"),  # 6 cm up at the tub is out of reach
    ]
    return {"right": right}
