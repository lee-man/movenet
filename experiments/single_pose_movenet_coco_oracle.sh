cd src
python main.py single_pose --exp_id coac_movenet_o --dataset active_coco --dataset active_coco --arch movenet --batch_size 24 --master_batch 4 --lr 2.5e-4 --gpus 0,1,2,3 --num_epochs 150 --lr_step 120 --num_workers 16 --load_model ../models/ctdet_movenet.pth --eval_oracle_offset --eval_oracle_wh
# test
python test.py single_pose --exp_id coac_movenet_o --dataset active_coco --arch movenet --keep_res --resume
# flip test
python test.py single_pose --exp_id coac_movenet_o --dataset active_coco --arch movenet --keep_res --resume --flip_test
cd ..

# demo
# python demo.py single_pose --dataset active_coco --arch movenet --demo ../images --load_model ../models/single_pose_movenet.pth --K 1 --gpus -1 --vis_thresh 0.0 --not_reg_offset