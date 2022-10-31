DATA_DIR="/data_cuda_ssd/minsu/109_cj_optical_flow/fold2/test/"
SAVE_DESCRIPT="histo_fold2_109"
SAVE_DIR="/data_cuda_ssd/minsu/0419_new_cj_binary_cross_validation/${SAVE_DESCRIPT}"
python main_histogram_svm.py --lr=0.00002 --aug=1.0 --epoch=50 --batch_size=3 --batch_size_test=1 --image_T=1 --optim=sgd --mode=2 --reverse_axis="" \
                             --cpus=16 --gpu=0 --resume=True --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --clip_frames=499 \
            	             --model=vivit_2 --description="Binary | Fold 1 | 109 | flow" --flow=True --test
