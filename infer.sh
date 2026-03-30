# export WANDB_HTTP_PROXY=http://192.168.1.182:7890
# export WANDB_HTTPS_PROXY=http://192.168.1.182:7890

python ./inference.py \
    --task_name move_pen \
    --use_robot_base True \
    --ckpt_dir /home/agilex/cobot_magic/origin_act_v2/move_pen_0330/  \
    --ckpt_name /home/agilex/cobot_magic/origin_act_v2/move_pen_0330/policy_last.ckpt \
    --state_dim 17 

# python ./act/inference.py \
#     --task_name move_water \
#     --ckpt_dir /home/agilex/cobot_magic/aloha-devel/arm_only_ckpt  \
#     --ckpt_name /home/agilex/cobot_magic/aloha-devel/arm_only_ckpt/policy_best.ckpt \
#     --state_dim 14 

    # --use_robot_base False \