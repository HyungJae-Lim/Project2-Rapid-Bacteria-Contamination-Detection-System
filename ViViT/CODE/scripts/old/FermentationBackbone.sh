DATA_DIR="../Fermentation_DATA/ver2"

SAVE_DESCRIPT="16X16_4clip_2.00margin"
SAVE_DIR="./outs/FermentationBackbone/${SAVE_DESCRIPT}"
python train_backbone.py --lr=0.00002 --aug=1.0 --epoch=50 --batch_size=100 --batch_size_test=100 --image_T=2 \
                         --optim=sgd --mode=1 --clip_frames=4 --cpus=16 --gpu=0 --resume="" --patch_size=16 \
                         --model=vivit_2 --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --backbone_path="" --margin=2.00 \
                         --description="Fermentation Backbone | 4 CLIP | 16X16 | margin=2.00"
