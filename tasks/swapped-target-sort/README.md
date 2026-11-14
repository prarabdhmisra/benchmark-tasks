# Put the red cube in the bin when a turntable swaps the cubes as the hand comes down

A motorized turntable (24 cm across) stands between the arms, carrying a red and a blue wooden cube 12 cm apart, red
on the robot's left. An open grey bin stands in front of the right arm. The robot is told: "Put the red cube in the
bin." When a gripper first comes down into the space over the turntable (a light curtain 12 cm above its top), the
turntable spins half a turn in 1.5 s, so the cubes trade places.

The instruction names the object by colour, and the object changes place after the robot has looked and started to
reach. A policy that re-reads the scene grasps the red cube at its new place. One that keeps acting on what it saw
first closes on the blue cube where the red one used to be, and puts the wrong item away. Picking stations,
pharmacies and sample racks all share that hazard: trays get rotated, carousels index, and items get rearranged
between a worker's look and their grasp. This task measures whether a policy re-grounds an instruction in the scene
as it changes, not in a snapshot. Frozen-scene sorting tasks can't measure it.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the light curtain and the turn (an `on_step` hook), `success()`, `score()`, `stale_grasp()` and the scripted `demo()` |
| `partkit.py` | Rigid parts built from boxes and cylinders, with step and reset hooks. Unchanged copy of the helper in `tasks/dual-latch-lid` |
| `test_task.py` | Checks of the scoring, the bin outline, the curtain, the turn's geometry and the demo's reach, without the simulator |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the red cube in the bin." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/sort-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the red cube in the bin." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/sort-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/sort-*/*.json; do inspect-robots video "$log"; done

# Scoring checks, no simulator needed
python -m pytest -q test_task.py
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the last step.

## Scripted demo

Right arm only. It moves over the red cube and comes down until its fingertips are 1 cm into the light curtain,
which starts the turn. It waits there, well clear of the cubes, until the turn has finished. Then it picks the red cube
from its new place (where the blue cube started) and drops it into the bin from 1 cm above the floor. The demo knows
where the red cube ends up, as a teleoperator watching the scene would; it only shows that a full score fits in the
run limit.

## Scoring

Evaluated every control step; the score at the last step counts (`SCORE_OVER = "end"`).

- **0** if the blue cube is ever in the bin, whatever happens afterwards.
- **1.0** (success, which must hold for 1 s): the red cube is in the bin: its centre within the bin's inner outline
  less half a cube and 5 mm (3.5 cm of the bin's centre along each axis), and below the bin's rim.
- **0.4**: the red cube was lifted at least 3 cm off the turntable, but is not in the bin.
- **0**: otherwise.

In the real world, these are judged from a photo from above and a ruler.

**Additional measure: stale grasp** (`stale_grasp()`, reported separately, not part of the score). The blue cube
was lifted 3 cm before the red one ever was. This is the specific failure the task is built to expose: grasping what
now occupies the place where the target used to be.

## Runtime and seeds

A full Dream-1 trial is 1,200 steps (1 minute at 20 Hz), plus Isaac Sim start-up. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

Rigid bodies. Physics runs at 240 Hz and control at 20 Hz. The turn is modelled by setting the disc's and both
cubes' poses and velocities every control step along a smoothstep profile (half a turn in 1.5 s); while it turns, the
cubes are held on the disc and the turntable does not stall against contact. The light curtain is the gripper
fingertips' computed positions (the tool point less the pad length) entering a 24 cm cylinder 12 cm tall over the
disc. No contact-force cap is enforced.

## Real-world build

- Turntable: a 24 cm, 2 cm thick disc on a geared stepper (a motorized lazy susan or a photography turntable),
  programmed for a smooth half turn in 1.5 s, current-limited so it stalls rather than pushes.
- Trigger: a pair of crossing break-beam sensors 12 cm above the disc. The turn starts the first time either beam is
  broken.
- Cubes: 4 cm painted wooden cubes, 100 g, one red, one blue, 6 cm either side of the axis. Bin: open-top PLA or
  plywood box, 12 x 12 cm inside, 6 cm tall, 5 mm walls.
