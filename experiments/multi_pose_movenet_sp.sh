cd src
python main.py multi_pose --exp_id cocosp_movenet --dataset coco_hpsp --arch movenet --batch_size 128 --master_batch 4 --lr 5e-4 --gpus 0,1,2,3 --num_epochs 320 --lr_step 270,300 --num_workers 16 --load_model ../models/ctdet_movenet.pth --eval_oracle_offset --eval_oracle_wh

# test
python test.py multi_pose --exp_id cocosp_movenet --dataset coco_hpsp --arch movenet --keep_res --resume
# flip test
python test.py multi_pose --exp_id cocosp_movenet --dataset coco_hpsp --arch movenet --keep_res --resume --flip_test
cd ..

# python demo.py multi_pose --dataset coco_hpsp --arch movenet --demo ../images --load_model ../models/model_best.pth --K 1 --gpus -1 --vis_thresh 0.0
