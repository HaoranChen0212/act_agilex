import unittest
from types import SimpleNamespace
from unittest import mock

import utils_xyyaw


class LoadDataSplitTest(unittest.TestCase):
    def setUp(self):
        self.dataset_dirs = ["dir_a", "dir_b", "dir_c"]
        self.dataset_lists = [
            ["dir_a/episode_0.hdf5", "dir_a/episode_1.hdf5"],
            ["dir_b/episode_0.hdf5", "dir_b/episode_1.hdf5"],
            ["dir_c/episode_0.hdf5", "dir_c/episode_1.hdf5"],
        ]
        self.norm_stats = {
            "qpos_mean": 0,
            "qpos_std": 1,
            "action_mean": 0,
            "action_std": 1,
        }
        self.relative_action_config = {"arm_active": False, "base_active": True, "chunk_size": 2}

    def _make_dataset(self, datalist, *args, **kwargs):
        return SimpleNamespace(datalist=list(datalist), is_sim=False)

    def _make_dataloader(self, dataset, *args, **kwargs):
        return SimpleNamespace(dataset=dataset)

    def test_load_data_can_mix_validation_across_all_dirs(self):
        with mock.patch.object(utils_xyyaw, "find_all_hdf5", side_effect=self.dataset_lists), \
             mock.patch.object(utils_xyyaw.random, "shuffle", side_effect=lambda items: None), \
             mock.patch.object(utils_xyyaw, "get_norm_stats", return_value=self.norm_stats), \
             mock.patch.object(utils_xyyaw, "EpisodicDataset", side_effect=self._make_dataset), \
             mock.patch.object(utils_xyyaw, "PadLastBatchCollator", return_value=None), \
             mock.patch.object(utils_xyyaw, "DataLoader", side_effect=self._make_dataloader):
            train_loader, val_loader, _, _ = utils_xyyaw.load_data(
                self.dataset_dirs,
                num_episodes=None,
                arm_delay_time=0,
                use_depth_image=False,
                use_robot_base=False,
                camera_names=["cam0"],
                batch_size_train=2,
                batch_size_val=2,
                relative_action_config=self.relative_action_config,
                use_all_dirs_for_val=True,
            )

        self.assertEqual(
            train_loader.dataset.datalist,
            [
                "dir_a/episode_0.hdf5",
                "dir_a/episode_1.hdf5",
                "dir_b/episode_0.hdf5",
                "dir_b/episode_1.hdf5",
                "dir_c/episode_0.hdf5",
            ],
        )
        self.assertEqual(val_loader.dataset.datalist, ["dir_c/episode_1.hdf5"])

    def test_load_data_keeps_legacy_first_dir_only_validation_when_disabled(self):
        with mock.patch.object(utils_xyyaw, "find_all_hdf5", side_effect=self.dataset_lists), \
             mock.patch.object(utils_xyyaw.random, "shuffle", side_effect=lambda items: None), \
             mock.patch.object(utils_xyyaw, "get_norm_stats", return_value=self.norm_stats), \
             mock.patch.object(utils_xyyaw, "EpisodicDataset", side_effect=self._make_dataset), \
             mock.patch.object(utils_xyyaw, "PadLastBatchCollator", return_value=None), \
             mock.patch.object(utils_xyyaw, "DataLoader", side_effect=self._make_dataloader):
            train_loader, val_loader, _, _ = utils_xyyaw.load_data(
                self.dataset_dirs,
                num_episodes=None,
                arm_delay_time=0,
                use_depth_image=False,
                use_robot_base=False,
                camera_names=["cam0"],
                batch_size_train=2,
                batch_size_val=2,
                relative_action_config=self.relative_action_config,
                use_all_dirs_for_val=False,
            )

        self.assertEqual(
            train_loader.dataset.datalist,
            [
                "dir_a/episode_0.hdf5",
                "dir_a/episode_0.hdf5",
                "dir_a/episode_0.hdf5",
                "dir_a/episode_0.hdf5",
                "dir_a/episode_0.hdf5",
                "dir_b/episode_0.hdf5",
                "dir_b/episode_1.hdf5",
                "dir_c/episode_0.hdf5",
                "dir_c/episode_1.hdf5",
            ],
        )
        self.assertEqual(val_loader.dataset.datalist, ["dir_a/episode_1.hdf5"])


if __name__ == "__main__":
    unittest.main()
