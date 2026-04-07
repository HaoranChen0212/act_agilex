#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange
from dataprocess import (
    CAMERA_NAMES as PANO_CAMERA_NAMES,
    DEFAULT_FOV as PANO_DEFAULT_FOV,
    DEFAULT_PATCH_SIZE as PANO_DEFAULT_PATCH_SIZE,
    build_view_specs as build_pano_view_specs,
    extract_view as extract_pano_view,
)

from utils_xyyaw import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
import collections
from collections import deque

from scipy.spatial.transform import Rotation

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import threading
import math
import threading
import cv2

import sys
sys.path.append("./")

DEFAULT_CAMERA_NAMES = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
CAMERA_NAME_TO_SOURCE = {
    'cam_high': 'front',
    'image_wide': 'front',
    'front': 'front',
    'cam_left_wrist': 'left',
    'left': 'left',
    'cam_right_wrist': 'right',
    'right': 'right',
}
PANO_CAMERA_NAME_SET = frozenset(PANO_CAMERA_NAMES)
PANO_VIEW_SPECS = tuple(build_pano_view_specs())

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None

base_init_qpos = None


def get_required_rgb_sources(camera_names):
    required_sources = set()
    for camera_name in camera_names:
        if camera_name in PANO_CAMERA_NAME_SET:
            required_sources.add('pano')
            continue

        source_name = CAMERA_NAME_TO_SOURCE.get(camera_name)
        if source_name is None:
            supported_names = ', '.join(sorted(set(CAMERA_NAME_TO_SOURCE) | PANO_CAMERA_NAME_SET))
            raise ValueError(
                f"Unsupported camera name '{camera_name}'. "
                f"Supported names: {supported_names}"
            )
        required_sources.add(source_name)
    return required_sources


def build_pano_camera_streams(
    panorama,
    camera_names,
    fov_deg=PANO_DEFAULT_FOV,
    patch_size=PANO_DEFAULT_PATCH_SIZE,
):
    view_spec_by_name = dict(PANO_VIEW_SPECS)
    pano_streams = {}
    for camera_name in camera_names:
        if camera_name not in view_spec_by_name:
            supported_names = ', '.join(PANO_CAMERA_NAMES)
            raise ValueError(
                f"Unsupported pano camera name '{camera_name}'. "
                f"Supported pano cameras: {supported_names}"
            )
        yaw_deg, pitch_deg = view_spec_by_name[camera_name]
        pano_streams[camera_name] = extract_pano_view(
            panorama=panorama,
            fov_deg=fov_deg,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            patch_size=patch_size,
        )
    return pano_streams


def build_preview_strip(images):
    images = [image for image in images if image is not None]
    if not images:
        raise ValueError("Expected at least one image to build a preview strip.")

    target_height = images[0].shape[0]
    resized_images = []
    for image in images:
        if image.shape[0] == target_height:
            resized_images.append(image)
            continue

        scale = target_height / float(image.shape[0])
        target_width = max(1, int(round(image.shape[1] * scale)))
        interpolation = cv2.INTER_AREA if image.shape[0] > target_height else cv2.INTER_LINEAR
        resized_images.append(
            cv2.resize(image, (target_width, target_height), interpolation=interpolation)
        )
    return np.hstack(resized_images)


def select_camera_streams(camera_names, front_value, left_value, right_value, pano_value=None):
    source_values = {
        'front': front_value,
        'left': left_value,
        'right': right_value,
    }
    selected = {}
    used_sources = set()
    requested_pano_cameras = [name for name in camera_names if name in PANO_CAMERA_NAME_SET]
    pano_streams = {}
    if requested_pano_cameras:
        panorama = pano_value if pano_value is not None else front_value
        if panorama is None:
            raise ValueError("Pano cameras were requested, but no pano image was provided.")
        pano_streams = build_pano_camera_streams(panorama, requested_pano_cameras)
    for camera_name in camera_names:
        if camera_name in pano_streams:
            selected[camera_name] = pano_streams[camera_name]
            continue
        source_name = CAMERA_NAME_TO_SOURCE.get(camera_name)
        if source_name is None:
            supported_names = ', '.join(sorted(set(CAMERA_NAME_TO_SOURCE) | PANO_CAMERA_NAME_SET))
            raise ValueError(
                f"Unsupported camera name '{camera_name}'. "
                f"Supported names: {supported_names}"
            )
        if source_name in used_sources:
            raise ValueError(
                f"Camera source '{source_name}' was selected more than once. "
                "Choose each available camera at most once."
            )
        selected[camera_name] = source_values[source_name]
        used_sources.add(source_name)
    return selected


def actions_interpolation(args, pre_action, actions, stats):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']
    result = [pre_action]
    post_action = post_process(actions[0])
    # print("pre_action:", pre_action[7:])
    # print("actions_interpolation1:", post_action[:, 7:])
    max_diff_index = 0
    max_diff = -1
    for i in range(post_action.shape[0]):
        diff = 0
        for j in range(pre_action.shape[0]):
            if j == 6 or j == 13:
                continue
            diff += math.fabs(pre_action[j] - post_action[i][j])
        if diff > max_diff:
            max_diff = diff
            max_diff_index = i

    for i in range(max_diff_index, post_action.shape[0]):
        step = max([math.floor(math.fabs(result[-1][j] - post_action[i][j])/steps[j]) for j in range(pre_action.shape[0])])
        inter = np.linspace(result[-1], post_action[i], step+2)
        result.extend(inter[1:])
    while len(result) < args.chunk_size+1:
        result.append(result[-1])
    result = np.array(result)[1:args.chunk_size+1]
    # print("actions_interpolation2:", result.shape, result[:, 7:])
    result = pre_process(result)
    result = result[np.newaxis, :]
    return result


def get_model_config(args):
    # 设置随机种子，你可以确保在相同的初始条件下，每次运行代码时生成的随机数序列是相同的。
    set_seed(1)
   
    # 如果是ACT策略
    # fixed parameters
    if args.policy_class == 'ACT':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': args.chunk_size,     # 查询
                         'camera_names': args.camera_names,
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base,
                         'kl_weight': args.kl_weight,        # kl散度权重
                         'hidden_dim': args.hidden_dim,      # 隐藏层维度
                         'dim_feedforward': args.dim_feedforward,
                         'enc_layers': args.enc_layers,
                         'dec_layers': args.dec_layers,
                         'nheads': args.nheads,
                         'dropout': args.dropout,
                         'pre_norm': args.pre_norm
                         }
    elif args.policy_class == 'CNNMLP':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': 1,     # 查询
                         'camera_names': args.camera_names,
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base
                         }

    elif args.policy_class == 'Diffusion':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': args.chunk_size,     # 查询
                         'camera_names': args.camera_names,
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base,
                         'observation_horizon': args.observation_horizon,
                         'action_horizon': args.action_horizon,
                         'num_inference_timesteps': args.num_inference_timesteps,
                         'ema_power': args.ema_power
                         }
    else:
        raise NotImplementedError

    config = {
        'ckpt_dir': args.ckpt_dir,
        'ckpt_name': args.ckpt_name,
        'ckpt_stats_name': args.ckpt_stats_name,
        'episode_len': args.max_publish_step,
        'state_dim': args.state_dim,
        'policy_class': args.policy_class,
        'policy_config': policy_config,
        'temporal_agg': args.temporal_agg,
        'camera_names': args.camera_names,
    }
    return config


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def get_depth_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_images.append(observation['images_depth'][cam_name])
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

from pose_util import compute_relative, xyzquat2mat, mat2xyzquat

    

def inference_process(args, config, ros_operator, policy, stats, t, pre_action):
    global inference_lock
    global inference_actions
    global inference_timestep
    global base_init_qpos
    
    print_flag = True
    pre_pos_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    pre_action_process = lambda next_action: (next_action - stats["action_mean"]) / stats["action_std"]
    rate = rospy.Rate(args.publish_rate)
    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, img_pano, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result
        
        bgr2rgb = True
        if bgr2rgb:
            if img_front is not None:
                img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)
            if img_left is not None:
                img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
            if img_right is not None:
                img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
            if img_pano is not None:
                img_pano = cv2.cvtColor(img_pano, cv2.COLOR_BGR2RGB)

        show_front = img_front if img_front is not None else img_pano
        show_left  = img_left
        show_right = img_right

        preview_images = []
        for image in (show_front, show_left, show_right):
            if image is None:
                continue
            if image.dtype != "uint8":
                image = (image * 255).astype("uint8")
            preview_images.append(image)
        if preview_images:
            view = build_preview_strip(preview_images)
            cv2.imwrite("/home/agilex/cobot_magic/aloha-devel/act/001.png", view)
        # xyz, quat = robot_base
        if args.use_robot_base: 
            if base_init_qpos is None:
                base_init_qpos = robot_base
            
            base_init_mat = xyzquat2mat(base_init_qpos[0], base_init_qpos[1])
            cur_mat = xyzquat2mat(robot_base[0], robot_base[1])
            rel_cur_mat = compute_relative(cur_mat, base_init_mat)
            xyz, quat = mat2xyzquat(rel_cur_mat)
            
            rpy = Rotation.from_quat(quat).as_euler('xyz', degrees=False)
            robot_base = xyz[0], xyz[1], rpy[-1] 
        
        obs = collections.OrderedDict()
        obs['images'] = select_camera_streams(
            config['camera_names'],
            front_value=img_front,
            left_value=img_left,
            right_value=img_right,
            pano_value=img_pano,
        )

        if args.use_depth_image:
            obs['images_depth'] = select_camera_streams(
                config['camera_names'],
                front_value=img_front_depth,
                left_value=img_left_depth,
                right_value=img_right_depth,
            )

        obs['qpos'] = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        obs['qvel'] = np.concatenate(
            (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
        obs['effort'] = np.concatenate(
            (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
        if args.use_robot_base:
            obs['base_vel'] = robot_base # [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]
        # qpos_numpy = np.array(obs['qpos'])

        # 归一化处理qpos 并转到cuda
        qpos = pre_pos_process(obs['qpos'])
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        # 当前图像curr_image获取图像
        curr_image = get_image(obs, config['camera_names'])
        curr_depth_image = None
        if args.use_depth_image:
            curr_depth_image = get_depth_image(obs, config['camera_names'])
            
        print('cur qpos:', obs['qpos'])
        start_time = time.time()
        all_actions = policy(curr_image, depth_image=None, robot_state=qpos,)
        end_time = time.time()
        print("model cost time: ", end_time -start_time)
        inference_lock.acquire()
        inference_actions = all_actions.cpu().detach().numpy()
        if pre_action is None:
            pre_action = obs['qpos']
        # print("obs['qpos']:", obs['qpos'][7:])
        if args.use_actions_interpolation:
            inference_actions = actions_interpolation(args, pre_action, inference_actions, stats)
        inference_timestep = t
        inference_lock.release()
        
        
        break


def model_inference(args, config, ros_operator, save_episode=True):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    set_seed(1000)

    # 1 创建模型数据  继承nn.
    policy = make_policy(config['policy_class'], config['policy_config'])
    # print("model structure\n", policy.model)
    
    # 2 加载模型权重
    ckpt_path = os.path.join(config['ckpt_dir'], config['ckpt_name'])
    new_state_dict = torch.load(ckpt_path)
    # new_state_dict = {}
    # for key, value in state_dict.items():
    #     if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
    #         continue
    #     if key in ["model.input_proj_next_action.weight", "model.input_proj_next_action.bias"]:
    #         continue
        # new_state_dict[key] = value
    loading_status = policy.deserialize(new_state_dict)
    if not loading_status:
        print("ckpt path not exist")
        return False

    # 3 模型设置为cuda模式和验证模式
    policy.cuda()
    policy.eval()

    # 4 加载统计值
    stats_path = os.path.join(config['ckpt_dir'], config['ckpt_stats_name'])
    # 统计的数据  # 加载action_mean, action_std, qpos_mean, qpos_std 14维
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 数据预处理和后处理函数定义
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']
    post_action_process = lambda next_action: next_action*stats["action_std"] + stats["action_mean"]

    max_publish_step = config['episode_len']
    chunk_size = config['policy_config']['chunk_size']

    # 发布基础的姿态
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    left0 = [0, 0.75, -0.9, 0.0, 0.7, 0]
    right0 = [0, 0.75, -0.9, 0.0, 0.7, 0]
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Enter any key to continue :")
    action = None
    # 推理
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            # 每个回合的步数
            t = 0
            max_t = 0
            rate = rospy.Rate(args.publish_rate)
            if config['temporal_agg']:
                all_time_actions = np.zeros([max_publish_step, max_publish_step + chunk_size, config['state_dim']])
            while t < max_publish_step and not rospy.is_shutdown():
                start_time = time.time()
                # query policy
                if config['policy_class'] == "ACT":
                    if t >= max_t:
                        pre_action = action
                        inference_thread = threading.Thread(target=inference_process,
                                                            args=(args, config, ros_operator,
                                                                  policy, stats, t, pre_action))
                        inference_thread.start()
                        inference_thread.join()
                        inference_lock.acquire()
                        if inference_actions is not None:
                            inference_thread = None
                            all_actions = inference_actions
                            inference_actions = None
                            max_t = t + args.pos_lookahead_step
                            if config['temporal_agg']:
                                all_time_actions[[t], t:t + chunk_size] = all_actions
                        inference_lock.release()
                    if config['temporal_agg']:
                        actions_for_curr_step = all_time_actions[:, t + 5]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        if args.pos_lookahead_step != 0:
                            raw_action = all_actions[:, t % args.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, t % chunk_size]
                else:
                    raise NotImplementedError
               
                action = post_action_process(raw_action[0])
                left_action = action[:7]  # 取7维度
                right_action = action[7:14]
                print(action)
                if t % 100 == 0:
                    import ipdb; ipdb.set_trace()
                
                ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread
                if args.use_robot_base:
                    base_dim = args.robot_base_action_dim
                    vel_action = action[14:14 + base_dim]
                    ros_operator.robot_base_publish(vel_action)
                t += 1
                end_time = time.time()
                print("publish: ", t)
                print("time:", end_time - start_time)
                print("left_action:", left_action)
                print("right_action:", right_action)
                rate.sleep()

class BaseController():
    def __init__(self, mode=None):
        self.mode = mode if mode else 'position'
        assert self.mode in ['velocity', 'position']
        self.kpxy = 0.8
        self.kpr = 0.5
        self.mode_continue = 0
        self.continue_thresh = 10
        self.position_mode = None
        
        self.wx = 1.0
        self.wy = 1.0
        self.wr = 1.0
    
    def snap_axis_for_slid_xy(self, vx, vy):
        ratio = 0.12
        threshold = max(ratio * abs(vy), 0.08)
        if abs(vx) < threshold:
            print('snap: vx {} -> 0, vy {}'.format(vx, vy))
        vx = 0 if abs(vx) <= threshold else vx
        return vx, vy
    
    def clamp(self, x, low, high):
        return max(low, min(x, high))

    def step(self, vel):
        vx, vy, vw = 0, 0, 0
        if self.mode == 'position':
            dx, dy, dyaw = vel
            xyzError = [dx, dy]
            rotError = dyaw
            
            if abs(rotError) < math.radians(5) and abs(xyzError[0]) + abs(xyzError[1]) < 0.08:
                print("No Need to Move")
                return 0, 0, 0

            slid_x = xyzError[0] * self.kpxy
            slid_y = xyzError[1] * self.kpxy
            Vr = self.kpr * rotError
            # wrap rotError into [-pi, pi]
            rotError = (rotError + np.pi) % (2 * np.pi) - np.pi
            print('xyzError: {}/{}, rotError: {}'.format(round(xyzError[0], 3), round(xyzError[1], 3), round(rotError, 3)))

            # ----- 4) Calculate Control Mode Cost -----
            if self.mode_continue > self.continue_thresh or self.position_mode is None:
                # C0 self rotation mode 
                C0 = self.wx * (slid_x) ** 2 + self.wy * (slid_y) ** 2 
                # C1 xy movement
                C1 = self.wr * (Vr) ** 2
                #C2 Alkamn 
                C2 = 1e5
                self.position_mode = [C0, C1, C2].index(min([C0, C1, C2]))
                print("C0: {}, C1: {}, C2: {}, mode_cnt".format(C0, C1, C2, self.mode_continue))
                self.mode_continue = 0
            else:
                self.mode_continue += 1
            # ----- 5) Send command based on mode -----
            if self.position_mode == 0:
                Vr = self.clamp(Vr, -0.5, 0.5)
                vx= 0
                vy = 0
                vw = Vr
                print('Select Mode: 0, linear_x = 0.0, linear_y = 0.0, angular_z: VR: {}'.format(round(Vr,3)))
            else: # self.position_mode == 1
                slid_x, slid_y = self.snap_axis_for_slid_xy(slid_x, slid_y)
                slid_x = self.clamp(slid_x, -0.6, 0.6)
                slid_y = self.clamp(slid_y, -0.6, 0.6)
                vx = slid_x
                vy = slid_y
                vw = 0
                print('Select Mode: 1, slid_x = {}, slid_y = {}, angular_z:0.0, VR: {}'.format(round(slid_x, 3), round(slid_y, 3), round(Vr,3)))
            
        else:
            raise NotImplementedError

        return vx, vy, vw

class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_pano_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        # self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.base_controller = None
        self.base_robot = None
        self.required_rgb_sources = get_required_rgb_sources(args.camera_names)
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_pano_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()
        
        

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        if len(vel) == 2:
            vx = vel[0]
            vy = 0
            w = vel[1]
        elif len(vel) == 3:
            if self.base_controller is None:
                self.base_controller = BaseController()
            # dx, dy, dyaw = vel
            print('vel:', vel)
            vx, vy, w = self.base_controller.step(vel)
        print('moveing ', vx, vy, w)
        self.base_robot.move(vx, vy, w)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def set_base_robot(self, base):
        self.base_robot = base 
    
    def get_frame(self):
        rgb_deque_by_source = {
            'front': self.img_front_deque,
            'left': self.img_left_deque,
            'right': self.img_right_deque,
            'pano': self.img_pano_deque,
        }
        if any(len(rgb_deque_by_source[source]) == 0 for source in self.required_rgb_sources):
            print('img err')
            return False
        if self.args.use_depth_image and (
            len(self.img_front_depth_deque) == 0 or len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0
        ):
            print('depth err')
            return False
        if len(self.puppet_arm_left_deque) == 0 or len(self.puppet_arm_right_deque) == 0:
            print('arm err')
            return False
        pending_queue = [rgb_deque_by_source[source] for source in sorted(self.required_rgb_sources)]
        
        
        
        # pick a sync time (same spirit as your code)
        if self.args.use_depth_image:
            pending_queue.extend([self.img_front_depth_deque, self.img_left_depth_deque, self.img_right_depth_deque])
            
        all_frame_time = [ a[-1].header.stamp.to_sec() for a in pending_queue ]
        frame_time = min(all_frame_time)


        # pop to sync
        def pop_to_time(dq):
            while len(dq) > 0 and dq[0].header.stamp.to_sec() < frame_time:
                dq.popleft()
            return dq.popleft() if len(dq) > 0 else None

        rgb_messages = {
            source: pop_to_time(rgb_deque_by_source[source])
            for source in sorted(self.required_rgb_sources)
        }
        m_larm = pop_to_time(self.puppet_arm_left_deque)
        m_rarm = pop_to_time(self.puppet_arm_right_deque)

        if any(message is None for message in rgb_messages.values()) or m_larm is None or m_rarm is None:
            return False

        img_front = None
        if 'front' in rgb_messages:
            img_front = self.bridge.imgmsg_to_cv2(rgb_messages['front'], "passthrough")
        img_left = None
        if 'left' in rgb_messages:
            img_left = self.bridge.imgmsg_to_cv2(rgb_messages['left'], "passthrough")
        img_right = None
        if 'right' in rgb_messages:
            img_right = self.bridge.imgmsg_to_cv2(rgb_messages['right'], "passthrough")
        img_pano = None
        if 'pano' in rgb_messages:
            img_pano = self.bridge.imgmsg_to_cv2(rgb_messages['pano'], "passthrough")

        img_front_depth = img_left_depth = img_right_depth = None
        if self.args.use_depth_image:
            md_front = pop_to_time(self.img_front_depth_deque)
            md_left = pop_to_time(self.img_left_depth_deque)
            md_right = pop_to_time(self.img_right_depth_deque)
            if md_front is None or md_left is None or md_right is None:
                return False
            img_front_depth = self.bridge.imgmsg_to_cv2(md_front, "passthrough")
            img_left_depth = self.bridge.imgmsg_to_cv2(md_left, "passthrough")
            img_right_depth = self.bridge.imgmsg_to_cv2(md_right, "passthrough")
            
        robot_base = None
        if self.args.use_robot_base:
            robot_base = self.base_robot.get_localization()

        return (
            img_front, img_left, img_right, img_pano,
            img_front_depth, img_left_depth, img_right_depth,
            m_larm, m_rarm, robot_base
        )

    def get_frame_old(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        
        
        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            # print('left:', self.img_left_deque[-1].header.stamp.to_sec(),'frame t:', frame_time)
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            # print('right:', self.img_right_deque[-1].header.stamp.to_sec(),'frame t:', frame_time)
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            # print('front:', self.img_front_deque[-1].header.stamp.to_sec(),'frame t:', frame_time)
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            # print('arm L:', self.puppet_arm_left_deque[-1].header.stamp.to_sec(),'frame t:', frame_time)
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            # print('arm R:', self.puppet_arm_right_deque[-1].header.stamp.to_sec(),'frame t:', frame_time)
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        # if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
        #     return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            # while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
            #     self.robot_base_deque.popleft()
                # robot_base = self.robot_base_deque.popleft()
            robot_base = self.base_robot.get_localization()
            # robot_base_twist = self.base_robot.get_twist()
        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_pano_callback(self, msg):
        if len(self.img_pano_deque) >= 2000:
            self.img_pano_deque.popleft()
        self.img_pano_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_pano_topic, Image, self.img_pano_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        # self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    parser.add_argument('--camera_names', action='store', nargs="+", type=str,
                        help='camera names used for inference image inputs',
                        default=DEFAULT_CAMERA_NAMES, required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/wide_cam/image_raw', required=False)
    parser.add_argument('--img_pano_topic', action='store', type=str, help='img_pano_topic',
                        default='/pano_cam/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--robot_base_action_dim', action='store', type=int, help='robot_base_action_dim',
                        default=3, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=40, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=32, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = get_model_config(args)
    
    if args.use_robot_base:
        from unimmdriver.adapters.mobile.rangermini import RangerMini, RangerMiniConfig
        cfg = RangerMiniConfig(
            http_url="http://192.168.1.102",
            ws_url="ws://192.168.1.102:9090",
            username="agilex",
            password_b64="NTdhMTE5NGNiMDczY2U4YjNiYjM2NWU0YjgwNWE5YWU=",
            frame_id="map_2d",
            heartbeat_interval_s=5.0,
        )
        ranger = RangerMini(cfg)
        ranger.connect()
        ranger.start_navigation("tong1208")
        ros_operator.set_base_robot(ranger)
    model_inference(args, config, ros_operator, save_episode=True)


if __name__ == '__main__':
    main()
# python act/inference.py --ckpt_dir ~/train0314/
