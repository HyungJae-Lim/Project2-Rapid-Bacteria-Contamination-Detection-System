DATA_DIR="/data/minsu/TWT_DATA/CFU"

SAVE_DESCRIPT="method1_fold2"
SAVE_DIR="/data/minsu/LOD_save/${SAVE_DESCRIPT}"
python main.py --lr=0.00002 --aug=1.0 --epoch=2 --batch_size=100 --batch_size_test=32 --image_T=1 --clip_frames=1 --optim=sgd --mode=1 \
               --patch_size=16 --cpus=8 --gpu=5 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" \
               --model="vivit_2" --description="fold2" --bacometer_val='Bacometer3' --bacometer_test='Bacometer4'
