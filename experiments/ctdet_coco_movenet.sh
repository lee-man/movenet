cd src
# train
python main.py ctdet --dataset coco --exp_id coco_movenet --arch movenet --batch_size 64 --master_batch 18 --lr 5e-4 --gpus 0,1,2,3 --num_workers 16 --resume
# test
python test.py ctdet --dataset coco --exp_id coco_movenet --arch movenet --keep_res --resume
# flip test
python test.py ctdet --dataset coco --exp_id coco_movenet --arch movenet --keep_res --resume --flip_test 
# multi scale test
python test.py ctdet --dataset coco --exp_id coco_movenet --arch movenet --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..

# python demo.py ctdet --dataset coco --arch movenet --demo ../images --load_model ../models/ctdet_movenet.pth --gpus -1 --vis_thresh 0.25 --nms