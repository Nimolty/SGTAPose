CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 train.py tracking \
                            --exp_id 1 \
                            --pre_hm \
                            --same_aug \
                            --hm_disturb 0.75 \
                            --lost_disturb 0.2 \
                            --fp_disturb 0.1 \
                            --gpus 5,6,7 \
                            --arch dlapawdl3new_34 \
                            --phase PlanA_win \
                            --dataset ../data/franka_data_1020 \
                            --add_dataset ../data/near_franka_data_1024 \
                            --val_dataset ../data/syn_test \
                            --is_real panda-3cam_realsense \
                            --num_epochs 20  \
                            --batch_size 4 \
