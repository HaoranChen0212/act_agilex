# Camera Names Configuration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the training camera set configurable from the CLI, then update the active training script to use `image_wide`, `cam_left_wrist`, and `cam_right_wrist` without breaking other datasets that do not contain `image_wide`.

**Architecture:** Add a `--camera_names` argument in `train.py`, use it instead of the hardcoded camera list inside `train()`, and keep the existing `cam_high` default for backward compatibility. Update `train.sh` to pass the requested wide-and-wrists camera list explicitly for the current run. Verify the change with a focused parser-and-wiring unit test that avoids running full training.

**Tech Stack:** Python, argparse, unittest, mock

---

### Task 1: Add a failing test for configurable camera names

**Files:**
- Create: `tests/test_train_camera_names.py`
- Modify: `train.py`

**Step 1: Write the failing test**

Write a `unittest` module that:
- asserts the default parsed camera list remains `["cam_high", "cam_left_wrist", "cam_right_wrist"]`
- asserts `train.train(args)` forwards parsed `camera_names` to `load_data`

**Step 2: Run test to verify it fails**

Run:

```bash
/home/haoran/miniconda3/envs/aloha/bin/python -m unittest tests.test_train_camera_names -v
```

Expected:
- failure because `train.py` does not yet define `--camera_names`
- or failure because `train()` still uses the hardcoded list

### Task 2: Implement configurable camera names

**Files:**
- Modify: `train.py`

**Step 1: Add the CLI argument**

Add:

```python
parser.add_argument(
    "--camera_names",
    action="store",
    nargs="+",
    type=str,
    default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
    help="camera names used for training inputs",
    required=False,
)
```

**Step 2: Remove the hardcoded task camera list**

Use `args.camera_names` inside `train()` when building task config and policy config.

**Step 3: Update the active training script**

Modify `train.sh` so the active `move_pen` run passes:

```bash
--camera_names image_wide cam_left_wrist cam_right_wrist
```

**Step 3: Keep behavior minimal**

Do not change dataset loading, model internals, or shell scripts unless needed. The existing stack already consumes `camera_names` dynamically.

### Task 3: Verify the implementation

**Files:**
- Modify: `tests/test_train_camera_names.py`
- Modify: `train.py`

**Step 1: Re-run the focused unit test**

Run:

```bash
/home/haoran/miniconda3/envs/aloha/bin/python -m unittest tests.test_train_camera_names -v
```

Expected:
- all tests pass

**Step 2: Sanity-check dataset compatibility**

Confirm the selected keys exist in the active `move_pen` dataset:
- `observations/images/image_wide`
- `observations/images/cam_left_wrist`
- `observations/images/cam_right_wrist`

**Step 3: Confirm the script is explicit**

Verify `train.sh` passes the requested camera list directly, so the run does not depend on parser defaults.
