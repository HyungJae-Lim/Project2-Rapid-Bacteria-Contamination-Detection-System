#DATA_DIR="../TEST6/train"
DATA_DIR="../TEST6/test/Contam"

SAVE_DESCRIPT="on_the_fly_axis_change"
SAVE_DIR="./outs/Fermentation_e2e/${SAVE_DESCRIPT}"
BACKBONE_MODEL="epoch[00432]_acc[0.5223]_test[0.0000].pth.tar"
BACKBONE_DIR="./outs/FermentationBackbone/${BACKBONE_MODEL}"

python train_cls_e2e.py --lr=0.00002 --aug=1.0 --epoch=1000 --batch_size=32 --batch_size_test=1 --image_T=1 --reverse_axis=True \
                         --optim=adam --mode=1 --clip_frames=1 --cpus=16 --gpu=1 --resume=True --patch_size=16 --margin=1.0 \
                         --model=vivit_2 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --backbone_path=${BACKBONE_DIR} --test \
                         --ClsNet_path="" --description="Fermentation End-to-End | on-the-fly | changed axis | 16X16"
