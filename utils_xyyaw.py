import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import random
import IPython

e = IPython.embed
import cv2
from pose_util import *


def compute_relative_base_action(pos, orientation, action_chunk_size):
    length = pos.shape[0]
    action_chunk_size = min(length, action_chunk_size)
    trajs = []
    for i in range(length - action_chunk_size + 1):
        p = pos[i]
        q = orientation[i]
        t_ref = xyzquat2mat(p, q)
        
        actions = zip(pos[i:i+action_chunk_size], orientation[i:i+action_chunk_size])
        # actions_rel = [mat2xyzquat(compute_relative(xyzquat2mat(*m), t_ref)) for m in actions]
        actions_rel = []
        for pp, qq in actions:
            t_cur = xyzquat2mat(pp, qq)
            t_rel = compute_relative(t_self=t_cur, t_ref=t_ref)
            t_rel = mat2xyzquat(t_rel)
            x,y = t_rel[0][0], t_rel[0][1]
            yaw = Rotation.from_quat(t_rel[1]).as_euler('xyz', degrees=False)[-1]
            actions_rel.append([x,y,yaw])
        trajs.append(actions_rel)
    return trajs
    
    
class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, datalist, camera_names, norm_stats, arm_delay_time,
                 use_depth_image, use_robot_base, horizon=64, relative_action_config={}):
        super(EpisodicDataset).__init__()
        self.datalist = datalist
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_depth_image = use_depth_image
        self.arm_delay_time = arm_delay_time
        self.use_robot_base = use_robot_base
        self.horizon = horizon
        self.relative_action_config = relative_action_config
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        dataset_path = self.datalist[index]
        base_active_relative = self.relative_action_config.get('base_active', False)
        
        H = self.horizon
        

        if base_active_relative:
            chunk_size = self.relative_action_config['chunk_size']
            H = chunk_size
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']

            # Episode length
            T = root['/action'].shape[0]   # 1501, etc.
            # We will use a fixed horizon H
            start_action = np.random.randint(self.arm_delay_time, T)

            # arm_delay_time offset
            index = start_action - self.arm_delay_time
            end = index + H                         # ensures action.shape[0] = H

            # ----- build actions from qpos (+ base) -----
            actions_all = root['/observations/qpos'][1:]    # (T, 14)
            actions_all = np.append(actions_all,
                                    actions_all[-1][np.newaxis, :],
                                    axis=0)                 # still ~ (T, 14)
            
            # TODO: index + 1 : end + 1 ??
            action = actions_all[index:end]                 # (H, 14)
            if '/observations/basePos' not in root:
                frame0_p, frame0_q = [0,0,0], [0,0,0,1] 
            else:
                frame0_p, frame0_q = root['/observations/basePos'][0], root['/observations/base_orientation'][0]
            t_frame0 = xyzquat2mat(frame0_p, frame0_q)

            if self.use_robot_base and '/observations/basePos' in root:
                if base_active_relative:
                    base_pos = root['/observations/basePos'][index:end]
                    base_orientation = root['/observations/base_orientation'][index:end]
                    rel_base_actions = compute_relative_base_action(base_pos, base_orientation, H)[0]
                    action = np.concatenate((action, rel_base_actions), axis=1)  # (H, 17)
                    # base = np.zeros_like(rel_base_actions)
                    # base[...,-1] = 1
                    # action = np.concatenate((action, base), axis=1)
                else:
                    # base_all = root['/base_action'][index:end]  # (H, 3)
                    base_pos = root['/observations/basePos'][index:end]  # (H, 3)
                    base_orientation = root['/observations/base_orientation'][index:end] #(H, 4)

                    action = np.concatenate((action, base_pos), axis=1)  # (H, 17)
                    action = np.concatenate((action, base_orientation), axis=1)  # (H, 21)
            else:
                action = np.concatenate((action, np.zeros([action.shape[0], 3])), axis=1)
            # No padding: all samples fixed shape (H, dim)
            action_len = H
            action_dim = action.shape[1]
            
            
            padded_action = np.zeros((action_len, action_dim), dtype=np.float32)
            padded_action[:action.shape[0]] = action
            action_is_pad = np.zeros(action_len)
            action_is_pad[action.shape[0]:] = 1

            # You still *can* keep action_is_pad if the model expects it:
            # now it's all real data → all zeros
            # action_is_pad = np.zeros(H, dtype=np.float32)

            # ----- single-step observation (as before) -----
            # choose qpos/image at the "current" time step
            qpos = root['/observations/qpos'][start_action]
            if self.use_robot_base and '/observations/basePos' in root:
                # qpos = np.concatenate((qpos, root['/base_action'][start_action]), axis=0)
                if base_active_relative:
                    base_start_p = root['/observations/basePos'][start_action]
                    base_start_q = root['/observations/base_orientation'][start_action]
                    t_cur = xyzquat2mat(base_start_p, base_start_q)
                    t_rel = compute_relative(t_self=t_cur, t_ref=t_frame0)
                    t_rel = mat2xyzquat(t_rel)
                    x,y = t_rel[0][0], t_rel[0][1]
                    yaw = Rotation.from_quat(t_rel[1]).as_euler('xyz', degrees=False)[-1]
                    base_qpos = [x,y,yaw]
                    qpos = np.concatenate((qpos, base_qpos), axis=0)
                else:
                    qpos = np.concatenate((qpos, root['/observations/basePos'][start_action]), axis=0)
                    qpos = np.concatenate((qpos, root['/observations/base_orientation'][start_action]), axis=0)
            else:
                qpos = np.concatenate((qpos, [0,0,0]), axis=0)
                
            image_dict = {}
            image_depth_dict = {}
            for cam_name in self.camera_names:
                if is_compress:
                    decoded_image = root[f'/observations/images/{cam_name}'][start_action]
                    image_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_action]

                if self.use_depth_image:
                    image_depth_dict[cam_name] = root[f'/observations/images_depth/{cam_name}'][start_action]

        # outside with: same post-processing as before
        self.is_sim = is_sim

        all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
        all_cam_images = np.stack(all_cam_images, axis=0)
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data) / 255.0

        image_depth_data = np.zeros(1, dtype=np.float32)
        if self.use_depth_image:
            all_cam_images_depth = [image_depth_dict[cam_name] for cam_name in self.camera_names]
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)
            image_depth_data = torch.from_numpy(all_cam_images_depth) / 255.0
        
        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        qpos_data = qpos_data.float()
        
        action_data = torch.from_numpy(padded_action).float()
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        action_data = action_data.float()
        
        action_is_pad = torch.from_numpy(action_is_pad).bool()
        return image_data, image_depth_data, qpos_data, action_data, action_is_pad, dataset_path
    
    def getitem__with_start(self, index, start):
        
        dataset_path = self.datalist[index]
        # if 'episode_45' in dataset_path:
        #     import ipdb; ipdb.set_trace()
        base_active_relative = self.relative_action_config.get('base_active', False)
        
        H = self.horizon
        

        if base_active_relative:
            chunk_size = self.relative_action_config['chunk_size']
            H = chunk_size
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']

            # Episode length
            T = root['/action'].shape[0]   # 1501, etc.
            # We will use a fixed horizon H
            start_action = start

            # arm_delay_time offset
            index = start_action - self.arm_delay_time
            end = index + H                         # ensures action.shape[0] = H

            # ----- build actions from qpos (+ base) -----
            actions_all = root['/observations/qpos'][1:]    # (T, 14)
            actions_all = np.append(actions_all,
                                    actions_all[-1][np.newaxis, :],
                                    axis=0)                 # still ~ (T, 14)
            
            # TODO: index + 1 : end + 1 ??
            action = actions_all[index:end]                 # (H, 14)
            frame0_p, frame0_q = root['/observations/basePos'][0], root['/observations/base_orientation'][0]
            t_frame0 = xyzquat2mat(frame0_p, frame0_q)

            if self.use_robot_base:
                if base_active_relative:
                    base_pos = root['/observations/basePos'][index:end]
                    base_orientation = root['/observations/base_orientation'][index:end]
                    rel_base_actions = compute_relative_base_action(base_pos, base_orientation, H)[0]
                    action = np.concatenate((action, rel_base_actions), axis=1)  # (H, 17)
                    # base = np.zeros_like(rel_base_actions)
                    # base[...,-1] = 1
                    # action = np.concatenate((action, base), axis=1)
                else:
                    # base_all = root['/base_action'][index:end]  # (H, 3)
                    base_pos = root['/observations/basePos'][index:end]  # (H, 3)
                    base_orientation = root['/observations/base_orientation'][index:end] #(H, 4)

                    action = np.concatenate((action, base_pos), axis=1)  # (H, 17)
                    action = np.concatenate((action, base_orientation), axis=1)  # (H, 21)
            
            # No padding: all samples fixed shape (H, dim)
            action_len = H
            action_dim = action.shape[1]
            
            
            padded_action = np.zeros((action_len, action_dim), dtype=np.float32)
            padded_action[:action.shape[0]] = action
            action_is_pad = np.zeros(action_len)
            action_is_pad[action.shape[0]:] = 1

            # You still *can* keep action_is_pad if the model expects it:
            # now it's all real data → all zeros
            # action_is_pad = np.zeros(H, dtype=np.float32)

            # ----- single-step observation (as before) -----
            # choose qpos/image at the "current" time step
            qpos = root['/observations/qpos'][start_action]
            if self.use_robot_base:
                # qpos = np.concatenate((qpos, root['/base_action'][start_action]), axis=0)
                if base_active_relative:
                    base_start_p = root['/observations/basePos'][start_action]
                    base_start_q = root['/observations/base_orientation'][start_action]
                    t_cur = xyzquat2mat(base_start_p, base_start_q)
                    t_rel = compute_relative(t_self=t_cur, t_ref=t_frame0)
                    t_rel = mat2xyzquat(t_rel)
                    x,y = t_rel[0][0], t_rel[0][1]
                    yaw = Rotation.from_quat(t_rel[1]).as_euler('xyz', degrees=False)[-1]
                    base_qpos = [x,y,yaw]
                    qpos = np.concatenate((qpos, base_qpos), axis=0)
                else:
                    qpos = np.concatenate((qpos, root['/observations/basePos'][start_action]), axis=0)
                    qpos = np.concatenate((qpos, root['/observations/base_orientation'][start_action]), axis=0)

            image_dict = {}
            image_depth_dict = {}
            for cam_name in self.camera_names:
                if is_compress:
                    decoded_image = root[f'/observations/images/{cam_name}'][start_action]
                    image_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_action]

                if self.use_depth_image:
                    image_depth_dict[cam_name] = root[f'/observations/images_depth/{cam_name}'][start_action]

        # outside with: same post-processing as before
        self.is_sim = is_sim

        all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
        all_cam_images = np.stack(all_cam_images, axis=0)
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data) / 255.0

        image_depth_data = np.zeros(1, dtype=np.float32)
        if self.use_depth_image:
            all_cam_images_depth = [image_depth_dict[cam_name] for cam_name in self.camera_names]
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)
            image_depth_data = torch.from_numpy(all_cam_images_depth) / 255.0
        
        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        qpos_data = qpos_data.float()
        
        action_data = torch.from_numpy(padded_action).float()
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        action_data = action_data.float()
        
        action_is_pad = torch.from_numpy(action_is_pad).bool()
        return image_data, image_depth_data, qpos_data, action_data, action_is_pad, dataset_path

    def __getitem_ori__(self, index):
        episode_id = self.episode_ids[index]
        # 读取数据
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']
            # print("action from file:", root['/action'].shape)
            # print("qpos:", root['/observations/qpos'].shape)
            # print("base_action:", root['/base_action'].shape)
            # input()
            original_action_shape = root['/action'].shape
            max_action_len = original_action_shape[0]  # max_episode
            if self.use_robot_base:
                original_action_shape = (original_action_shape[0], original_action_shape[1] + 3)

            start_ts = np.random.choice(max_action_len)  # 随机抽取一个索引
            actions = root['/observations/qpos'][1:]
            actions = np.append(actions, actions[-1][np.newaxis, :], axis=0)
            qpos = root['/observations/qpos'][start_ts]
            if self.use_robot_base:
                qpos = np.concatenate((qpos, root['/base_action'][start_ts]), axis=0)
            image_dict = dict()
            image_depth_dict = dict()
            for cam_name in self.camera_names:
                if is_compress:
                    decoded_image = root[f'/observations/images/{cam_name}'][start_ts]
                    image_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                    # print(image_dict[cam_name].shape)
                    # exit(-1)
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

                if self.use_depth_image:
                    image_depth_dict[cam_name] = root[f'/observations/images_depth/{cam_name}'][start_ts]

            start_action = min(start_ts, max_action_len - 1)
            index = max(0, start_action - self.arm_delay_time)
            action = actions[index:]  # hack, to make timesteps more aligned
            if self.use_robot_base:
                action = np.concatenate((action, root['/base_action'][index:]), axis=1)
            action_len = max_action_len - index  # hack, to make timesteps more aligned

        self.is_sim = is_sim

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        action_is_pad = np.zeros(max_action_len)
        action_is_pad[action_len:] = 1
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        
        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0

        image_depth_data = np.zeros(1, dtype=np.float32)
        if self.use_depth_image:
            all_cam_images_depth = []
            for cam_name in self.camera_names:
                all_cam_images_depth.append(image_depth_dict[cam_name])
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)
            # construct observations
            image_depth_data = torch.from_numpy(all_cam_images_depth)
            # image_depth_data = torch.einsum('k h w c -> k c h w', image_depth_data)
            image_depth_data = image_depth_data / 255.0

        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        qpos_data = qpos_data.float()
        action_data = torch.from_numpy(padded_action).float()
        action_is_pad = torch.from_numpy(action_is_pad).bool()
        action_data = (action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # torch.set_printoptions(precision=10, sci_mode=False)
        # torch.set_printoptions(threshold=float('inf'))
        # print("qpos_data:", qpos_data[7:])
        # print("action_data:", action_data[:, 7:])

        return image_data, image_depth_data, qpos_data, action_data, action_is_pad


    

def get_norm_stats(datalist, use_robot_base, relative_action_config):
    all_qpos_data = []
    all_action_data = []
    all_rel_base_action_data = [] # [episodes, L, chunk_size, action_dim]
    all_rel_base_qpos_data = []
    base_active_relative = relative_action_config['base_active']
    chunk_size = relative_action_config['chunk_size']
    total_frames = 0
    for dataset_path in datalist:
        with h5py.File(dataset_path, 'r') as root:
            if '/observations/qpos' not in root:
                print(dataset_path)
                
                
            qpos = root['/observations/qpos'][()]
            # qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
            length = qpos.shape[0]
            total_frames += length
            # the base action and qpos will be calc norm by all_rel_base_qpos_data and all_rel_base_action_data
            if use_robot_base and '/observations/basePos' in root:
                if base_active_relative:
                    rel_base_action_chunk = compute_relative_base_action(root['/observations/basePos'][()], root['/observations/base_orientation'][()], action_chunk_size=chunk_size)
                    all_rel_base_action_data.append(rel_base_action_chunk)
                    base_qpos = compute_relative_base_action(root['/observations/basePos'][()], root['/observations/base_orientation'][()], action_chunk_size=length)[0]
                    all_rel_base_qpos_data.append(base_qpos)
                    
                    qpos = np.concatenate((qpos, np.array(base_qpos)), axis=1)
                    action = np.concatenate((action, np.zeros([action.shape[0], 3])), axis=1) # mock dummy base data, not used for normalization
                else:
                    qpos = np.concatenate((qpos, root['/observations/basePos'][()]), axis=1)
                    qpos = np.concatenate((qpos, root['/observations/base_orientation'][()]), axis=1)
                    action = np.concatenate((action, root['/observations/basePos'][()]), axis=1)
                    action = np.concatenate((action, root['/observations/base_orientation'][()]), axis=1)
            else: # for the static cotraining data
                qpos = np.concatenate((qpos, np.zeros([qpos.shape[0], 3])), axis=1)
                action = np.concatenate((action, np.zeros([action.shape[0], 3])), axis=1) # mock dummy base data, not used for normalization
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    if len(set(d.shape[0] for d in all_qpos_data)) > 1:
        all_qpos_data = torch.cat(all_qpos_data).unsqueeze(0)
        all_action_data = torch.cat(all_action_data).unsqueeze(0)
    else:
        all_qpos_data = torch.stack(all_qpos_data)
        all_action_data = torch.stack(all_action_data)
    
    if base_active_relative:
        flat = [a for ep in all_rel_base_action_data for t in ep for a in t]
        rel_base_actions = torch.tensor(flat)
        rel_base_action_mean = rel_base_actions.mean(dim=0, keepdim=True)
        rel_base_action_std = rel_base_actions.std(dim=0, keepdim=True)
        rel_base_action_std = torch.clip(rel_base_action_std, 1e-2, np.inf) # 7
        
        flat = [a for ep in all_rel_base_qpos_data for a in ep]
        rel_base_qposs = torch.tensor(flat)
        rel_base_qposs_mean = rel_base_qposs.mean(dim=0, keepdim=True)
        rel_base_qposs_std = rel_base_qposs.std(dim=0, keepdim=True)
        rel_base_qposs_std = torch.clip(rel_base_qposs_std, 1e-2, np.inf) # 7

    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping
    action_mean_var = action_mean.numpy().squeeze()
    action_std_var  = action_std.numpy().squeeze()
    qpos_mean_var = qpos_mean.numpy().squeeze()
    qpos_std_var  = qpos_std.numpy().squeeze()
    if base_active_relative:
        action_mean_var[-3:] = rel_base_action_mean
        action_std_var[-3:] = rel_base_action_std
        qpos_mean_var[-3:] = rel_base_qposs_mean
        qpos_std_var[-3:] = rel_base_qposs_std
    stats = {"action_mean": action_mean_var, "action_std": action_std_var,
             "qpos_mean": qpos_mean_var, "qpos_std": qpos_std_var,
             "example_qpos": qpos}
    print('compute norm stat from: ', total_frames)
    return stats


class PadLastBatchCollator:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __call__(self, batch):
        cur_bs = len(batch)
        if cur_bs < self.batch_size:
            extra = random.choices(
                range(len(self.dataset)),
                k=self.batch_size - cur_bs
            )
            for idx in extra:
                batch.append(self.dataset[idx])

        return torch.utils.data._utils.collate.default_collate(batch)

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    import fnmatch
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def load_data(dataset_dir_l, num_episodes, arm_delay_time, use_depth_image,
              use_robot_base, camera_names, batch_size_train, batch_size_val,
              relative_action_config, use_all_dirs_for_val=False):
    print(f'\nData from: {dataset_dir_l}\n')
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    skip_mirrored_data = False
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    dataset_path_list = flatten_list(dataset_path_list_list)

    # obtain train test split
    train_ratio = 0.9
    if use_all_dirs_for_val:
        random.shuffle(dataset_path_list)
        split_idx = int(train_ratio * len(dataset_path_list))
        train_datalist = dataset_path_list[:split_idx]
        val_datalist = dataset_path_list[split_idx:]
    else:
        num_episodes_0 = len(dataset_path_list_list[0])
        ep0_datalist = dataset_path_list[0:num_episodes_0]
        other_datalist = dataset_path_list[num_episodes_0:]
        random.shuffle(ep0_datalist)
        train_datalist = ep0_datalist[0: int(train_ratio * num_episodes_0)] * 5 + other_datalist
        val_datalist = ep0_datalist[int(train_ratio * num_episodes_0):]

    print('val dataset list: ', val_datalist)
        

    # obtain normalization stats for qpos and action  返回均值和方差
    norm_stats = get_norm_stats(list(set(dataset_path_list)), use_robot_base, relative_action_config)

    # construct dataset and dataloader 归一化处理  结构化处理数据
    train_dataset = EpisodicDataset(train_datalist, camera_names, norm_stats, arm_delay_time,
                                    use_depth_image, use_robot_base, relative_action_config=relative_action_config)

    val_dataset = EpisodicDataset(val_datalist, camera_names, norm_stats, arm_delay_time,
                                  use_depth_image, use_robot_base, relative_action_config=relative_action_config)
    collate_fn = PadLastBatchCollator(train_dataset, batch_size_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, collate_fn=collate_fn,
                                  num_workers=4, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=2, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


# env utils
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])

    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


# helper functions
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
