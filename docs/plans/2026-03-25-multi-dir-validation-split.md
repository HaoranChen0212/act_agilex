# Multi-Directory Validation Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a training flag that optionally mixes all dataset directories into a shared train/validation split while preserving the current first-directory-only validation behavior by default.

**Architecture:** Extend `train.py` with a backward-compatible boolean argument, thread it through to `load_data`, and centralize the split decision inside `utils_xyyaw.load_data`. Cover both parser wiring and split semantics with focused unit tests before changing production code.

**Tech Stack:** Python, argparse, unittest, unittest.mock

---

### Task 1: Lock Parser And Forwarding Behavior

**Files:**
- Modify: `tests/test_train_camera_names.py`
- Modify: `train.py`

**Step 1: Write the failing test**

Assert that `get_arguments()` exposes `use_all_dirs_for_val` with a default of `False`, and that `train.train(args)` forwards a parsed `True` value into `load_data`.

**Step 2: Run test to verify it fails**

Run: `/home/haoran/miniconda3/envs/aloha/bin/python -m unittest tests.test_train_camera_names -v`

Expected: parser is missing the new attribute and rejects `--use_all_dirs_for_val`.

**Step 3: Write minimal implementation**

Add `--use_all_dirs_for_val` to `train.py` and pass it into `load_data(...)`.

**Step 4: Run test to verify it passes**

Run: `/home/haoran/miniconda3/envs/aloha/bin/python -m unittest tests.test_train_camera_names -v`

Expected: PASS.

### Task 2: Lock Split Semantics

**Files:**
- Create: `tests/test_load_data_split.py`
- Modify: `utils_xyyaw.py`

**Step 1: Write the failing test**

Add one test asserting `use_all_dirs_for_val=True` produces a global mixed split across all directories, and one test asserting `False` preserves the legacy first-directory-only validation behavior.

**Step 2: Run test to verify it fails**

Run: `/home/haoran/miniconda3/envs/aloha/bin/python -m unittest tests.test_load_data_split -v`

Expected: `load_data()` rejects the new argument.

**Step 3: Write minimal implementation**

Update `utils_xyyaw.load_data` to accept the new flag and branch the split logic accordingly.

**Step 4: Run test to verify it passes**

Run: `/home/haoran/miniconda3/envs/aloha/bin/python -m unittest tests.test_load_data_split -v`

Expected: PASS.

### Task 3: Verify Combined Behavior

**Files:**
- Modify: `train.py`
- Modify: `utils_xyyaw.py`
- Verify: `tests/test_train_camera_names.py`
- Verify: `tests/test_load_data_split.py`

**Step 1: Run focused regression tests**

Run: `/home/haoran/miniconda3/envs/aloha/bin/python -m unittest tests.test_train_camera_names tests.test_load_data_split -v`

Expected: all tests pass.
