import os
import sys
import unittest
from unittest import mock
from unittest.mock import mock_open


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import train


class TrainCameraNamesTest(unittest.TestCase):
    def test_default_camera_names_remain_backward_compatible(self):
        argv = [
            "train.py",
            "--dataset_dir",
            "dummy_dataset",
            "--ckpt_dir",
            "/tmp/ckpt",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = train.get_arguments()

        self.assertEqual(
            args.camera_names,
            ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        )
        self.assertFalse(args.use_all_dirs_for_val)

    def test_train_uses_configured_camera_names(self):
        argv = [
            "train.py",
            "--dataset_dir",
            "dummy_dataset",
            "dummy_dataset_extra",
            "--ckpt_dir",
            "/tmp/ckpt",
            "--camera_names",
            "image_wide",
            "cam_left_wrist",
            "cam_right_wrist",
            "--use_all_dirs_for_val",
            "True",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = train.get_arguments()

        captured = {}

        def fake_load_data(
            dataset_dir,
            num_episodes,
            arm_delay_time,
            use_depth_image,
            use_robot_base,
            camera_names,
            batch_size_train,
            batch_size_val,
            relative_action_config,
            use_all_dirs_for_val,
        ):
            captured["camera_names"] = camera_names
            captured["use_all_dirs_for_val"] = use_all_dirs_for_val
            return [], [], {"qpos_std": 1, "qpos_mean": 0, "action_mean": 0, "action_std": 1}, False

        with mock.patch.object(train, "load_data", side_effect=fake_load_data), \
             mock.patch.object(train, "train_process", return_value=(0, 0.0, {})), \
             mock.patch.object(train.os.path, "isdir", return_value=True), \
             mock.patch("builtins.open", mock_open()), \
             mock.patch.object(train.pickle, "dump"), \
             mock.patch.object(train.torch, "save"):
            train.train(args)

        self.assertEqual(
            captured["camera_names"],
            ["image_wide", "cam_left_wrist", "cam_right_wrist"],
        )
        self.assertTrue(captured["use_all_dirs_for_val"])


if __name__ == "__main__":
    unittest.main()
