# Put a lid on a tub that slides away once the lid is picked up

An open white tub (10 x 10 x 7 cm) is fixed to the carriage of a motorized linear stage in front of the right arm. A
round blue lid (17 cm across, with a 4.5 cm upright knob) lies flat on a switch pad to the tub's right. The robot is
told: "Put the blue lid on the white tub." Lifting the lid off its pad starts the stage: in 1 s the tub slides 8 cm
toward the robot's left and stops.

The scene changes after the robot has seen it and committed to a plan. A policy that re-reads the scene follows the
tub; one that keeps acting on what it saw before the move lowers the lid where the tub used to be. Packing and capping
lines, lab benches and kitchens all move containers mid-task: a conveyor indexes, or a person nudges a tub aside. This
task measures whether a policy notices that change before it acts. Frozen-scene tasks can't measure it.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the stage (moved by an `on_step` hook), `success()`, `score()`, `stale_placement()` and the scripted `demo()` |
| `partkit.py` | Rigid parts built from boxes and cylinders, with step and reset hooks. Unchanged copy of the helper in `tasks/dual-latch-lid` |
| `test_task.py` | Checks of the scoring bands, the knock rules, the stale-placement measure and the demo's reach, without the simulator |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the blue lid on the white tub." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tub-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the blue lid on the white tub." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/tub-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/tub-*/*.json; do inspect-robots video "$log"; done

# Scoring checks, no simulator needed
python -m pytest -q test_task.py
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the last step.

## Scripted demo

Right arm only. It grips the knob from above (fingertips 8 mm over the lid), lifts the lid, which starts the stage,
and waits at carry height until the tub has stopped. Then it carries the lid over the tub's new centre, lowers it to
4 mm over the rim, lets go, rises 3 cm and moves back toward its base. The demo knows where the stage stops, as a teleoperator watching the scene would;
it only shows that a full score fits in the run limit.

## Scoring

Evaluated every control step; the score at the last step counts (`SCORE_OVER = "end"`).

- **0** if the tub is knocked at any time: tilted more than 10 degrees, or pushed more than 2 cm from where it stood
  (its start before the move, its stop after it). The stage's own move is not a knock.
- **1.0** (success, which must hold for 1 s): after the move has finished, the lid rests on the tub's rim (underside
  between 3 mm below and 5 mm above the rim), tilted under 5 degrees, with its centre within 1.5 cm of the tub's centre
  seen from above. At that offset the 17 cm lid still covers the whole opening.
- **0.5**: the lid rests on or just over the tub off centre (centre within 5 cm, underside between 1 cm below and 2 cm
  above the rim), or tilted.
- **0.2**: the lid was lifted off its pad (the stage moved), but it is not on the tub.
- **0**: the lid was never lifted.

In the real world, every condition is measured with a ruler, a spirit level and a photo from above.

**Additional measure: stale placement** (`stale_placement()`, reported separately, not part of the score). After
the move, the lid ends on the stage bed or table with its centre within 6 cm of where the tub stood before the move.
This is the specific failure the task is built to expose: acting on stale perception.

## Runtime and seeds

The scripted demo finishes in about 12 s of simulated time and then holds the success for 1 s. A full Dream-1 trial is
1,200 steps (1 minute at 20 Hz), plus Isaac Sim start-up. The layout is fixed (`-E object_jitter=0`); there are no seeds
to choose.

## Fidelity

Rigid bodies. Physics runs at 240 Hz and control at 20 Hz. The stage is modelled by teleporting the tub's body every control step along a smoothstep
profile (8 cm in 1 s, peak 12 cm/s) and setting its velocity to match; while it moves, the tub is held upright and the
stage does not stall if something pushes against it. The trigger is the lid's underside rising 1 cm above its pad. The
tub is heavy (1.5 kg with its carriage) so it stays put once the stage stops. No contact-force cap is enforced.

## Real-world build

- Stage: any motorized linear stage with at least 10 cm of travel and a 12 cm wide bed (10 mm tall), such as a
  belt-driven rail with a stepper motor. Program it to move 8 cm in 1 s with smooth acceleration, then stop and hold.
  Run it current-limited so it stalls rather than pushes if a gripper blocks it.
- Trigger: a membrane switch pad (18 x 18 cm) under the lid. The stage starts when the lid's weight leaves it.
- Tub: a 10 x 10 x 7 cm open container (4 mm walls) screwed to a 14 x 14 x 1 cm carriage plate. Lid: a 17 cm disc,
  6 mm thick, with a 25 mm diameter, 45 mm tall knob at its centre (3D-printed PLA, about 80 g).
