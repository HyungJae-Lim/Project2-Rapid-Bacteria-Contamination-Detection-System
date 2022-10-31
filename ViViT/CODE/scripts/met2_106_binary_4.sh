DATA_DIR="/data/minsu/0419_Speckles/106"

SAVE_DESCRIPT="method2_fold4"
SAVE_DIR="/data/minsu/106_binary/${SAVE_DESCRIPT}"
python main_v2.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=4 --batch_size_test=1 --image_T=10 --clip_frames=50 --optim=sgd --mode=2 \
                  --patch_size=16 --cpus=8 --gpu=1 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" \
                  --model="vivit_2" --description="fold4" --method=2 --fold=4
