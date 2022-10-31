DATA_DIR="/data/minsu/0419_Speckles/106"

SAVE_DESCRIPT="method1_fold1"
SAVE_DIR="/data/minsu/106_binary/${SAVE_DESCRIPT}"
python main_v2.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=4 --batch_size_test=1 --image_T=10 --clip_frames=500 --optim=sgd --mode=2 \
                  --patch_size=16 --cpus=8 --gpu=3 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" \
                  --model="vivit_2" --description="fold1" --val_sample='34' --method=1

SAVE_DESCRIPT="method1_fold1"
SAVE_DIR="/data/minsu/106_binary/${SAVE_DESCRIPT}"
python main_v2.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=4 --batch_size_test=1 --image_T=10 --clip_frames=250 --optim=sgd --mode=2 \
                  --patch_size=16 --cpus=8 --gpu=3 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" \
                  --model="vivit_2" --description="fold1" --val_sample='34' --method=1

SAVE_DESCRIPT="method1_fold1"
SAVE_DIR="/data/minsu/106_binary/${SAVE_DESCRIPT}"
python main_v2.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=4 --batch_size_test=1 --image_T=10 --clip_frames=100 --optim=sgd --mode=2 \
                  --patch_size=16 --cpus=8 --gpu=3 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" \
                  --model="vivit_2" --description="fold1" --val_sample='34' --method=1

SAVE_DESCRIPT="method1_fold1"
SAVE_DIR="/data/minsu/106_binary/${SAVE_DESCRIPT}"
python main_v2.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=4 --batch_size_test=1 --image_T=10 --clip_frames=20 --optim=sgd --mode=2 \
                  --patch_size=16 --cpus=8 --gpu=3 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" \
                  --model="vivit_2" --description="fold1" --val_sample='34' --method=1

