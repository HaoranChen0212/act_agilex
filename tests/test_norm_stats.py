import math
import os
import tempfile
import unittest

import h5py
import numpy as np
import torch

from utils_xyyaw import EpisodicDataset, compute_relative_base_action, get_norm_stats


def yaw_to_quat(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], dtype=np.float32)


class NormStatsTest(unittest.TestCase):
    def test_relative_base_qpos_stats_are_computed_per_dimension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "episode_0.hdf5")

            qpos = np.zeros((4, 14), dtype=np.float32)
            action = np.zeros((4, 14), dtype=np.float32)
            base_pos = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.5, 2.0, 0.0],
                    [3.0, 2.5, 0.0],
                ],
                dtype=np.float32,
            )
            base_orientation = np.stack(
                [
                    yaw_to_quat(0.0),
                    yaw_to_quat(0.2),
                    yaw_to_quat(-0.4),
                    yaw_to_quat(0.7),
                ],
                axis=0,
            )

            with h5py.File(path, "w") as root:
                obs = root.create_group("observations")
                obs.create_dataset("qpos", data=qpos)
                obs.create_dataset("basePos", data=base_pos)
                obs.create_dataset("base_orientation", data=base_orientation)
                root.create_dataset("action", data=action)

            stats = get_norm_stats(
                [path],
                use_robot_base=True,
                relative_action_config={"arm_active": False, "base_active": True, "chunk_size": 2},
            )

            rel_base_qpos = np.asarray(
                compute_relative_base_action(base_pos, base_orientation, action_chunk_size=len(qpos))[0],
                dtype=np.float32,
            )
            expected_mean = torch.tensor(rel_base_qpos).mean(dim=0).numpy()
            expected_std = torch.tensor(rel_base_qpos).std(dim=0).clamp(1e-2, torch.inf).numpy()

            np.testing.assert_allclose(stats["qpos_mean"][-3:], expected_mean, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(stats["qpos_std"][-3:], expected_std, rtol=1e-6, atol=1e-6)

    def test_getitem_with_start_keeps_normalized_qpos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "episode_0.hdf5")

            qpos = np.array(
                [
                    np.arange(14, dtype=np.float32),
                    np.arange(14, dtype=np.float32) + 10,
                    np.arange(14, dtype=np.float32) + 20,
                    np.arange(14, dtype=np.float32) + 30,
                ],
                dtype=np.float32,
            )
            action = np.zeros((4, 14), dtype=np.float32)
            images = np.zeros((4, 2, 2, 3), dtype=np.uint8)
            base_pos = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.5, 0.0],
                    [1.5, 1.0, 0.0],
                    [2.0, 1.5, 0.0],
                ],
                dtype=np.float32,
            )
            base_orientation = np.stack(
                [
                    yaw_to_quat(0.0),
                    yaw_to_quat(0.2),
                    yaw_to_quat(0.4),
                    yaw_to_quat(0.6),
                ],
                axis=0,
            )

            with h5py.File(path, "w") as root:
                root.attrs["sim"] = False
                root.attrs["compress"] = False
                obs = root.create_group("observations")
                obs.create_dataset("qpos", data=qpos)
                obs.create_dataset("basePos", data=base_pos)
                obs.create_dataset("base_orientation", data=base_orientation)
                obs.create_dataset("images/cam0", data=images)
                root.create_dataset("action", data=action)

            norm_stats = {
                "qpos_mean": np.ones(17, dtype=np.float32),
                "qpos_std": np.full(17, 2.0, dtype=np.float32),
                "action_mean": np.zeros(17, dtype=np.float32),
                "action_std": np.ones(17, dtype=np.float32),
            }
            dataset = EpisodicDataset(
                [path],
                ["cam0"],
                norm_stats,
                arm_delay_time=0,
                use_depth_image=False,
                use_robot_base=True,
                relative_action_config={"arm_active": False, "base_active": True, "chunk_size": 2},
            )

            _, _, qpos_data, _, _, _ = dataset.getitem__with_start(0, 1)
            rel_base_qpos = np.asarray(
                compute_relative_base_action(base_pos, base_orientation, action_chunk_size=len(qpos))[0][1],
                dtype=np.float32,
            )
            expected_qpos = np.concatenate([qpos[1], rel_base_qpos], axis=0)
            expected = (expected_qpos - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]

            np.testing.assert_allclose(qpos_data.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_getitem_returns_float32_qpos_after_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "episode_0.hdf5")

            qpos = np.array(
                [
                    np.arange(14, dtype=np.float32),
                    np.arange(14, dtype=np.float32) + 1,
                ],
                dtype=np.float32,
            )
            action = np.zeros((2, 14), dtype=np.float32)
            images = np.zeros((2, 2, 2, 3), dtype=np.uint8)

            with h5py.File(path, "w") as root:
                root.attrs["sim"] = False
                root.attrs["compress"] = False
                obs = root.create_group("observations")
                obs.create_dataset("qpos", data=qpos)
                obs.create_dataset("images/cam0", data=images)
                root.create_dataset("action", data=action)

            dataset = EpisodicDataset(
                [path],
                ["cam0"],
                {
                    "qpos_mean": np.zeros(17, dtype=np.float64),
                    "qpos_std": np.ones(17, dtype=np.float64),
                    "action_mean": np.zeros(17, dtype=np.float64),
                    "action_std": np.ones(17, dtype=np.float64),
                },
                arm_delay_time=0,
                use_depth_image=False,
                use_robot_base=False,
                relative_action_config={"arm_active": False, "base_active": True, "chunk_size": 2},
            )

            _, _, qpos_data, _, _, _ = dataset[0]

            self.assertEqual(qpos_data.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
