# finetune scripts for movenet
cd src
python main.py single_pose --exp_id yoga_movenet --dataset active --arch movenet --batch_size 24 --master_batch 4 --lr 5e-4 --gpus 0,1,2,3 --num_epochs 250 --lr_step 120,150,180,200,230  --num_workers 16 --eval_oracle_offset --eval_oracle_wh --load_model ../models/movenet.pth 
# test
# python test.py single_pose --exp_id yoga_movenet --dataset active --arch movenet --keep_res --resume
# # flip test
# python test.py single_pose --exp_id yoga_movenet --dataset active --arch movenet --keep_res --resume --flip_test
cd ..

# python demo.py single_pose --dataset active_coco --arch movenet --demo ../images/active --load_model ../models/single_pose_active.pth --K 1 --gpus -1 --vis_thresh 0.0 --not_reg_offset
