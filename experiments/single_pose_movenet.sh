cd src
python main.py single_pose --exp_id coco_movenet --dataset active --arch movenet --batch_size 64 --master_batch 4 --lr 5e-4 --gpus 0,1,2,3 --num_epochs 320 --lr_step 270,300 --num_workers 16 --load_model ../models/ctdet_movenet.pth --eval_oracle_offset --eval_oracle_wh

# test
python test.py single_pose --exp_id coco_movenet --dataset active --arch movenet --keep_res --resume
# flip test
python test.py single_pose --exp_id coco_movenet --dataset active --arch movenet --keep_res --resume --flip_test
cd ..

# python demo.py single_pose --dataset active_coco --arch movenet --demo ../images --load_model ../models/model_last.pth --K 1 --gpus -1 --vis_thresh 0.0
