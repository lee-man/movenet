cd src
python main.py multi_pose --exp_id cocosp_movenet --dataset coco_hpsp --arch movenet --batch_size 24 --master_batch 4 --lr 2.5e-4 --gpus 0,1,2,3 --num_epochs 50 --lr_step 40 --num_workers 16 --resume
# test
python test.py multi_pose --exp_id cocosp_movenet --dataset coco_hpsp --arch movenet --keep_res --resume
# flip test
python test.py multi_pose --exp_id cocosp_movenet --dataset coco_hpsp --arch movenet --keep_res --resume --flip_test
cd ..