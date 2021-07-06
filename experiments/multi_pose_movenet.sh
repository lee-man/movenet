cd src
python main.py multi_pose --exp_id movenet --dataset coco_hp --arch mobilenet --batch_size 24 --master_batch 4 --lr 2.5e-4 --gpus 0,1,2,3 --num_epochs 50 --lr_step 40