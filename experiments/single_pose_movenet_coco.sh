cd src
python main.py single_pose --exp_id coac_movenet --dataset active_coco --dataset active_coco --arch movenet --batch_size 24 --master_batch 4 --lr 2.5e-4 --gpus 0,1,2,3 --num_epochs 250 --lr_step 120,150,180,200,230 --num_workers 16 --resume #--load_model ../models/ctdet_movenet.pth
# test
python test.py single_pose --exp_id coac_movenet --dataset active_coco --arch movenet --keep_res --resume
# flip test
python test.py single_pose --exp_id coac_movenet --dataset active_coco --arch movenet --keep_res --resume --flip_test
cd ..