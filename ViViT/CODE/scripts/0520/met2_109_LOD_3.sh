DATA_DIR="/data/minsu/0419_Speckles/109"

SAVE_DESCRIPT="met2_109_fold3"
SAVE_DIR="/data/minsu/LOD_109_save/${SAVE_DESCRIPT}"
python main.py --lr=0.00002 --aug=1.0 --epoch=50 --batch_size=100 --batch_size_test=32 --image_T=1 --clip_frames=1 --optim=sgd --mode=2 \
               --patch_size=16 --cpus=8 --gpu=2 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --fold=3 \
               --model="vivit_2" --description="fold3" --bacometer_val='cam_0' --bacometer_test='Bacometer5' \
               --val_sample='29'
