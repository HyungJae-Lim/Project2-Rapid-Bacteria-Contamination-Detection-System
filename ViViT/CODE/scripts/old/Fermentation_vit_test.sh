DATA_DIR="../TEST1/test/Uncontam"
#DATA_DIR="../TEST1/test/Contam/Contam1"
#DATA_DIR="../TEST1/test/Contam/Contam10"
#DATA_DIR="../TEST1/test/Contam/Contam25"
#DATA_DIR="../TEST1/test/Contam/Contam50"

SAVE_DESCRIPT="experiment1_32batch"
SAVE_DIR="./outs/fermentation_vit_experiment/${SAVE_DESCRIPT}"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=32 --image_T=1 --optim=sgd --mode=1 \
               --cpus=16 --gpu=0 --resume=True --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --clip_frames=1 \
               --model=vivit_2 --description="Limit Of Detection, 64 patch | 1T" --test

