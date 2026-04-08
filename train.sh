# export WANDB_HTTP_PROXY=http://192.168.1.182:7890
# export WANDB_HTTPS_PROXY=http://192.168.1.182:7890

# python train.py --dataset_dir move_water /home/haoran/coTrainning/aloha_static_cotraining_datasets \
#     --use_robot_base True --ckpt_dir ~/exp/train_move_water_e49_cos_lr_base_joint_rel_xyyaw_cotrain/  \
#     --batch_size 16 --num_epochs 4000 \
#     --exp_name train_move_water_e49_v2_cos_lr_base_joint_rel_xyyaw_fix_decoder_cotrain


# python train.py --dataset_dir data/move_pen  \
#     --use_robot_base True --ckpt_dir /exp/move_pen_rotate_place/  \
#     --camera_names image_wide cam_left_wrist cam_right_wrist \
#     --batch_size 16 --num_epochs 10000 \
#     --exp_name act_pick_pen_routate_place



# python train.py \
#     --dataset_dir data/move_pen_v1 data/move_pen_v2 data/move_pen_v3 \
#     --camera_names image_wide cam_right_wrist \
#     --use_all_dirs_for_val True \
#     --use_robot_base True \
#     --exp_name act_pick_pen_mix_123 \
#     --ckpt_dir exp/move_pen_mix_123 \
#     --batch_size 16 \
#     --num_epochs 20000
    
# python train.py \
#     --dataset_dir ../arrange_chair_v2 \
#     --camera_names pano_yaw_m180 pano_yaw_m120 pano_yaw_m060 pano_yaw_p000 pano_yaw_p060 pano_yaw_p120 \
#     --use_all_dirs_for_val True \
#     --use_robot_base True \
#     --exp_name act_arrange_chair_widenwrist \
#     --ckpt_dir exp/arrange_chair_widenwrist \
#     --batch_size 16 \
#     --num_epochs 20000


python train.py \
    --dataset_dir ../move_pen/move_pen_9im_v1 ../move_pen/move_pen_9im_v2 ../move_pen/move_pen_9im_v3 \
    --camera_names pano_yaw_m180 pano_yaw_m120 pano_yaw_m060 pano_yaw_p000 pano_yaw_p060 pano_yaw_p120 image_wide cam_right_wrist\
    --use_all_dirs_for_val True \
    --use_robot_base True \
    --exp_name move_pen_pano_as_8im \
    --ckpt_dir ../exp/move_pen_pano_as_8im \
    --batch_size 8 \
    --num_epochs 25000
