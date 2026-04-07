# export WANDB_HTTP_PROXY=http://192.168.1.182:7890
# export WANDB_HTTPS_PROXY=http://192.168.1.182:7890

CAMERA_NAMES=(cam_high cam_left_wrist cam_right_wrist)
# Match these to the checkpoint's training camera order.
# Example for the local 2-camera move_pen_mix_123 checkpoint:
# CAMERA_NAMES=(image_wide cam_right_wrist)
CAMERA_NAMES=(pano_yaw_m180 pano_yaw_m120 pano_yaw_m060 pano_yaw_p000 pano_yaw_p060 pano_yaw_p120)
PANO_TOPIC=/pano_cam/image_raw

python ./inference.py \
    --task_name move_pen \
    --use_robot_base True \
    --ckpt_dir /home/agilex/cobot_magic/origin_act_v2/move_pen_0330/  \
    --ckpt_name /home/agilex/cobot_magic/origin_act_v2/move_pen_0330/policy_last.ckpt \
    --img_pano_topic "${PANO_TOPIC}" \
    --camera_names "${CAMERA_NAMES[@]}" \
    --state_dim 17 

# python ./act/inference.py \
#     --task_name move_water \
#     --ckpt_dir /home/agilex/cobot_magic/aloha-devel/arm_only_ckpt  \
#     --ckpt_name /home/agilex/cobot_magic/aloha-devel/arm_only_ckpt/policy_best.ckpt \
#     --state_dim 14 

    # --use_robot_base False \
