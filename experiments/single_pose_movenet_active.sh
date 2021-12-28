# finetune scripts for movenet
cd /Users/rachel/PycharmProjects/movenet/src
# python main.py single_pose --exp_id yoga_movenet --dataset active --arch movenet --batch_size 24 --master_batch 4 --lr 5e-4 --gpus 0,1,2,3 --num_epochs 150 --lr_step 30 60 90 --num_workers 16 --load_model ../models/movenet.pth
# test   7e -3   5e-3  1e-3  5e-4  1e-4
# python test.py single_pose --exp_id yoga_movenet --dataset active --arch movenet --resume
# # flip test
# python test.py single_pose --exp_id yoga_movenet --dataset active --arch movenet --resume --flip_test
#python demo.py single_pose --dataset active_pair --arch movenet --demo ../images/1124visual/ --load_model ../models/movenet_thunder.pth --K 1 --gpus -1 --debug 2 --vis_thresh 0.0 #--not_reg_offset

# 1202only_kpt_0.005_2.pth
python demo.py single_pose --dataset aist_add --arch movenet --demo ../data/add_aist/val --load_model ../models/1213add_aist_aug_0.6_90_0.005.pth --K 1 --gpus -1 --debug 2 --vis_thresh 0.1 #--not_reg_offset

#python demo.py single_pose --dataset active_pair --arch movenet --demo ../experiments/1123test/left_ankle/ --load_model ../models/movenet_thunder.pth --K 1 --gpus -1 --debug 2 --vis_thresh 0.0 #--not_reg_offset

#python demo.py single_pose --dataset active_pair --arch movenet --demo ../experiments/1123test/left_ankle/ --load_model ../models/1120_only_no_center_0.005.pth --K 1 --gpus -1 --debug 2 --vis_thresh 0.0 #--not_reg_offset

cd ..
# 原版和aug效果最差的后400张