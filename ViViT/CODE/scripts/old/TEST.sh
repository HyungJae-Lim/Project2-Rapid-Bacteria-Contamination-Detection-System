DATA_DIR="../DATA/SPECIES"

SAVE_DESCRIPT="2T"
SAVE_DIR="./outs/vivit/SPECIES/${SAVE_DESCRIPT}"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=32 --image_T=2 --optim=sgd --mode=2 \
               --cpus=4 --gpu=1 --resume=True --test --blind_test="" \
               --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" \
               --description="2T"

SAVE_DESCRIPT="30T"
SAVE_DIR="./outs/vivit/SPECIES/${SAVE_DESCRIPT}"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=32 --image_T=30 --optim=sgd --mode=2 \
               --cpus=4 --gpu=1 --resume=True --test --blind_test="" \
               --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" \
               --description="30T"

SAVE_DESCRIPT="60T"
SAVE_DIR="./outs/vivit/SPECIES/${SAVE_DESCRIPT}"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=32 --image_T=60 --optim=sgd --mode=2 \
               --cpus=4 --gpu=1 --resume=True --test --blind_test="" \
               --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" \
               --description="60T"

