SAVE_DESCRIPT="fold1_aug"
SAVE_DIR="/data_cuda_ssd/minsu/vit_logs/vivit/${SAVE_DESCRIPT}"
DATA_DIR="/data_cuda_ssd/minsu/vivit_1st/train/"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=15 --batch_size_test=1 --image_T=1 --flow="" \
                         --optim=sgd --mode=1 --clip_frames=1 --cpus=8 --gpu=1 --resume="" --patch_size=16 \
                         --model=vivit_2 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --mode=0 \
                         --description="Fermentation classification | fold1 | augmentation"

SAVE_DESCRIPT="fold2_aug"
SAVE_DIR="/data_cuda_ssd/minsu/vit_logs/vivit/${SAVE_DESCRIPT}"
DATA_DIR="/data_cuda_ssd/minsu/vivit_2nd/train/"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=15 --batch_size_test=1 --image_T=1 --flow="" \
                         --optim=sgd --mode=1 --clip_frames=1 --cpus=8 --gpu=1 --resume="" --patch_size=16 \
                         --model=vivit_2 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --mode=0 \
                         --description="Fermentation classification | fold2 | augmentation"

SAVE_DESCRIPT="fold3_aug"
SAVE_DIR="/data_cuda_ssd/minsu/vit_logs/vivit/${SAVE_DESCRIPT}"
DATA_DIR="/data_cuda_ssd/minsu/vivit_3rd/train/"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=15 --batch_size_test=1 --image_T=1 --flow="" \
                         --optim=sgd --mode=1 --clip_frames=1 --cpus=8 --gpu=1 --resume="" --patch_size=16 \
                         --model=vivit_2 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --mode=0 \
                         --description="Fermentation classification | fold3 | augmentation"

SAVE_DESCRIPT="fold4_aug"
SAVE_DIR="/data_cuda_ssd/minsu/vit_logs/vivit/${SAVE_DESCRIPT}"
DATA_DIR="/data_cuda_ssd/minsu/vivit_4th/train/"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=15 --batch_size_test=1 --image_T=1 --flow="" \
                         --optim=sgd --mode=1 --clip_frames=1 --cpus=8 --gpu=1 --resume="" --patch_size=16 \
                         --model=vivit_2 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --mode=0 \
                         --description="Fermentation classification | fold4 | augmentation"

SAVE_DESCRIPT="fold5_aug"
SAVE_DIR="/data_cuda_ssd/minsu/vit_logs/vivit/${SAVE_DESCRIPT}"
DATA_DIR="/data_cuda_ssd/minsu/vivit_5th/train/"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=15 --batch_size_test=1 --image_T=1 --flow="" \
                         --optim=sgd --mode=1 --clip_frames=1 --cpus=8 --gpu=1 --resume="" --patch_size=16 \
                         --model=vivit_2 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --mode=0 \
                         --description="Fermentation classification | fold5 | augmentation"

SAVE_DESCRIPT="fold6_aug"
SAVE_DIR="/data_cuda_ssd/minsu/vit_logs/vivit/${SAVE_DESCRIPT}"
DATA_DIR="/data_cuda_ssd/minsu/vivit_6th/train/"
python main.py --lr=0.00002 --aug=1.0 --epoch=100 --batch_size=15 --batch_size_test=1 --image_T=1 --flow="" \
                         --optim=sgd --mode=1 --clip_frames=1 --cpus=8 --gpu=1 --resume="" --patch_size=16 \
                         --model=vivit_2 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --mode=0 \
                         --description="Fermentation classification | fold6 | augmentation"
