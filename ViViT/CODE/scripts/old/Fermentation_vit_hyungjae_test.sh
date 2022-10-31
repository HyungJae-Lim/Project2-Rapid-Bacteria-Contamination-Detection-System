DATA_DIR="../TEST_hyungjae/test/"
SAVE_DESCRIPT="hyungjae_data_settings"
SAVE_DIR="./outs/vit/${SAVE_DESCRIPT}"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=30 --batch_size_test=1 --image_T=1 --optim=sgd --mode=0 --reverse_axis="" \
               --cpus=16 --gpu=2 --resume=True --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --clip_frames=1 --test \
               --model=vivit_2 --description="1~6 train sample | 7~12 test sample"
