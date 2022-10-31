DATA_DIR="../Fermentation_DATA/ver2"

SAVE_DESCRIPT="ver2_16patch"
SAVE_DIR="./outs/vivit/Fermentation_normal"
python main.py --lr=0.00002 --aug=1.0 --epoch=999 --batch_size=32 --image_T=2 --optim=sgd --mode=1 --patch_size=16 \
               --cpus=16 --gpu=2 --resume="" --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --clip_frames=4 \
               --model=vivit_2 --description="Normal Fermentation | data version 2 | 32 batch | 16 patch | 4 clip frames"

