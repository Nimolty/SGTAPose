CUDA_VISIBLE_DEVICES=0 python inference.py tracking \
                              --real_info_path ../dream_real_info \
                              --infer_dataset ../data \
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


CUDA_VISIBLE_DEVICES=0 python inference.py tracking \
                              --real_info_path ../dream_real_info \
                              --infer_dataset ../data \
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
                              --is_real panda-3cam_azure \
                              --rf 

CUDA_VISIBLE_DEVICES=0 python inference.py tracking \
                              --real_info_path ../dream_real_info \
                              --infer_dataset ../data \
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
                              --is_real panda-3cam_realsense \
                              --rf                               





