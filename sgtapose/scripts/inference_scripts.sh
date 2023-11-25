#real_tests="1_D415_front_0 2_D415_front_1 3_kinect_front_0 4_kinect_front_1 6_D415_left_1 8_kinect_left_1 9_kinect_left_0 10_kinect_right_1 12_D415_right_1"
#epoch_lists="10 11 12 13 14 15 16 17 18 19 20"
#for real_test in $real_tests
#do
#  for epoch in $epoch_lists
#  do
#    CUDA_VISIBLE_DEVICES=6 python Dream_ct_inference_ours.py tracking --load_model /DATA/disk1/hyperplane/robot_pose_3090/center-dream/tracking/0820_1/ckpt/model_$epoch.pth  --exp_id 0820_1 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawdl3new_34 --phase PlanA_win  --is_real $real_test --depth_dataset "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/" --rf --kps_name dream_7
#  done
#done
# real_tests="panda-3cam_azure panda-orb"
# epoch_lists="10 11 12 13 14 15 16 17 18 19 20"
# for real_test in $real_tests
# do
#   for epoch in $epoch_lists
#   do
#     CUDA_VISIBLE_DEVICES=6 python Dream_ct_inference.py tracking --load_model /DATA/disk1/hyperplane/robot_pose_3090/center-dream/tracking/0820_1/ckpt/model_$epoch.pth  --exp_id 0820_1 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawdl3new_34 --phase PlanA_win  --is_real $real_test --depth_dataset "/DATA/disk1/hyperplane/ty_data/" --rf
#   done
# done
#CUDA_VISIBLE_DEVICES=6 python Dream_ct_inference.py tracking --load_model /DATA/disk1/hyperplane/robot_pose_3090/center-dream/tracking/0820_1/ckpt/model_20.pth  --exp_id 0820_1 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawdl3new_34 --phase PlanA_win  --is_real panda-3cam_realsense --depth_dataset "/DATA/disk1/hyperplane/ty_data/" --rf

#CUDA_VISIBLE_DEVICES=0 python Dream_ct_inference.py tracking --load_model "/DATA/disk1/hyperplane/robot_pose_3090/center-dream/tracking/0820_1/ckpt/model_20.pth"  --exp_id 0820_1 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawdl3new_34 --phase PlanA_win  --is_real 1_D415_front_0 --depth_dataset "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/" --rf
#CUDA_VISIBLE_DEVICES=0 python Dream_ct_inference.py tracking --load_model "/DATA/disk1/hyperplane/robot_pose_3090/center-dream/tracking/0820_1/ckpt/model_20.pth"  --exp_id 0820_1 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawdl3new_34 --phase PlanA_win  --is_real 2_D415_front_1 --depth_dataset "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/" --rf
#CUDA_VISIBLE_DEVICES=0 python Dream_ct_inference.py tracking --load_model "/DATA/disk1/hyperplane/robot_pose_3090/center-dream/tracking/0820_1/ckpt/model_20.pth"  --exp_id 0820_1 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawdl3new_34 --phase PlanA_win  --is_real 3_kinect_front_0 --depth_dataset "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/" --rf
#CUDA_VISIBLE_DEVICES=0 python Dream_ct_inference.py tracking --load_model "/DATA/disk1/hyperplane/robot_pose_3090/center-dream/tracking/0820_1/ckpt/model_20.pth"  --exp_id 0820_1 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawdl3new_34 --phase PlanA_win  --is_real 4_kinect_front_1 --depth_dataset "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/" --rf
#CUDA_VISIBLE_DEVICES=0 python Dream_ct_inference.py tracking --load_model "/DATA/disk1/hyperplane/robot_pose_3090/center-dream/tracking/0820_1/ckpt/model_20.pth"  --exp_id 0820_1 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawdl3new_34 --phase PlanA_win  --is_real 6_D415_left_1 --depth_dataset "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/" --rf
#CUDA_VISIBLE_DEVICES=0 python Dream_ct_inference.py tracking --load_model "/DATA/disk1/hyperplane/robot_pose_3090/center-dream/tracking/0820_1/ckpt/model_20.pth"  --exp_id 0820_1 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawdl3new_34 --phase PlanA_win  --is_real 8_kinect_left_1 --depth_dataset "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/" --rf
#CUDA_VISIBLE_DEVICES=0 python Dream_ct_inference.py tracking --load_model "/DATA/disk1/hyperplane/robot_pose_3090/center-dream/tracking/0820_1/ckpt/model_20.pth"  --exp_id 0820_1 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawdl3new_34 --phase PlanA_win  --is_real 9_kinect_left_0 --depth_dataset "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/" --rf
#CUDA_VISIBLE_DEVICES=0 python Dream_ct_inference.py tracking --load_model "/DATA/disk1/hyperplane/robot_pose_3090/center-dream/tracking/0820_1/ckpt/model_20.pth"  --exp_id 0820_1 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawdl3new_34 --phase PlanA_win  --is_real 10_kinect_right_1 --depth_dataset "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/" --rf
#CUDA_VISIBLE_DEVICES=0 python Dream_ct_inference.py tracking --load_model "/DATA/disk1/hyperplane/robot_pose_3090/center-dream/tracking/0820_1/ckpt/model_20.pth"  --exp_id 0820_1 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawdl3new_34 --phase PlanA_win  --is_real 12_D415_right_1 --depth_dataset "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/" --rf



CUDA_VISIBLE_DEVICES=0 python inference.py tracking \
                              --real_info_path ../dream_real_info \
                              --infer_dataset /DATA/disk1/hyperplane/ty_data \
                              --load_model ../pretrained_model/ckpt/model_20.pth  \
                              --pre_hm \
                              --same_aug \
                              --hm_disturb 0.75 \
                              --lost_disturb 0.2 \
                              --fp_disturb 0.1 \
                              --gpus 3 \
                              --root_dir "../result" \
                              --arch dlapawdl3new_34 \
                              --phase PlanA_win  \
                              --is_real panda-orb \
                              --rf 








