DATA_DIR="/data/minsu/0419_Speckles/109"

SAVE_DESCRIPT="method1_109_fold1"
SAVE_DIR="/data/minsu/LOD_109_save/${SAVE_DESCRIPT}"
python main.py --lr=0.00002 --aug=1.0 --epoch=5 --batch_size=100 --batch_size_test=32 --image_T=1 --clip_frames=1 --optim=sgd --mode=1 \
               --patch_size=16 --cpus=8 --gpu=0 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --fold=1 \
               --model="vivit_2" --description="fold1" --bacometer_val='Bacometer4' --bacometer_test='Bacometer5' \
               --val_sample='25'
