DATA_DIR="/data/minsu/0419_Speckles/106"

SAVE_DESCRIPT="method1_106_fold5"
SAVE_DIR="/data/minsu/LOD_106_save/${SAVE_DESCRIPT}"
python main.py --lr=0.00002 --aug=1.0 --epoch=5 --batch_size=100 --batch_size_test=32 --image_T=1 --clip_frames=1 --optim=sgd --mode=2 \
               --patch_size=16 --cpus=8 --gpu=3 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --fold=5 \
               --model="vivit_2" --description="fold5" --bacometer_val='Bacometer5' --bacometer_test='Bacometer1' \
               --val_sample='26'
