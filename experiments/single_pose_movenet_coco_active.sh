cd src
python main.py single_pose --exp_id coac_movenet --dataset active_coco --dataset active_coco --arch movenet --batch_size 64 --master_batch 4 --lr 5e-4 --gpus 0,1,2,3 --num_epochs 320 --lr_step 270,300 --num_workers 16 --eval_oracle_offset --eval_oracle_wh --load_model ../models/ctdet_movenet.pth
# test
python test.py single_pose --exp_id coac_movenet --dataset active_coco --arch movenet --keep_res --resume
# flip test
python test.py single_pose --exp_id coac_movenet --dataset active_coco --arch movenet --keep_res --resume --flip_test
cd ..