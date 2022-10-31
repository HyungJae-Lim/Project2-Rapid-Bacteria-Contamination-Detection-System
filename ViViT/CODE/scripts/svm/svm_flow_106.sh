DATA_DIR="/data_cuda_ssd/minsu/106_cj_optical_flow_0429/fold1/train/"
SAVE_DESCRIPT="histo_fold1_106"
SAVE_DIR="/data_cuda_ssd/minsu/0429_new_cj_binary_cross_validation/${SAVE_DESCRIPT}"
python main_histogram_svm.py --lr=0.00002 --aug=1.0 --epoch=50 --batch_size=3 --batch_size_test=1 --image_T=1 --optim=sgd --mode=2 --reverse_axis="" \
                             --cpus=16 --gpu=0 --resume="" --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --clip_frames=499 \
            	             --model=vivit_2 --description="Binary | Fold 1 | 109 | flow" --flow=True
