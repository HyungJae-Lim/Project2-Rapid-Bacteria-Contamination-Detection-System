DATA_DIR="../TEST6/train"
SAVE_DESCRIPT="experiment6_2labels"
SAVE_DIR="./outs/fermentation_vit/${SAVE_DESCRIPT}"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=32 --image_T=1 --optim=sgd --mode=1 --reverse_axis=True \
               --cpus=16 --gpu=0 --resume="" --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --clip_frames=1 \
               --model=vivit_2 --description="Experiment 6 | axis permute | 2 labels"
