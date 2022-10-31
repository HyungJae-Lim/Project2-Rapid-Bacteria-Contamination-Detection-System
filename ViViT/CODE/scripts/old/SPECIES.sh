DATA_DIR="/data/minsu/TWT_DATA/CFU/"

SAVE_DESCRIPT="TEST"
SAVE_DIR="/data/minsu/SPECIES_save/${SAVE_DESCRIPT}"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=16 --image_T=1 --clip_frames=1 --optim=sgd --mode=0 \
               --patch_size=16 --cpus=8 --gpu=0 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" \
               --model="vivit_2" --description="Species Classification, 300FPS | 64 patch | 300T"
