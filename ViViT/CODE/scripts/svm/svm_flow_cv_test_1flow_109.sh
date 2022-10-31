DATA_DIR="/data_cuda_ssd/minsu/109_cj_optical_flow/fold1/test/"
SAVE_DESCRIPT="histo_fold1_per_1flow_109"
SAVE_DIR="/data_cuda_ssd/minsu/0419_new_cj_binary_cross_validation/${SAVE_DESCRIPT}"
python main_histogram_svm.py --lr=0.00002 --aug=1.0 --epoch=50 --batch_size=1 --batch_size_test=1 --image_T=1 --optim=sgd --mode=2 --reverse_axis="" \
                             --cpus=16 --gpu=0 --resume=True --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --clip_frames=1 \
                             --model=vivit_2 --description="Binary | Fold 1 | 109 | 20 batch | 1 clip" --flow=True --test

DATA_DIR="/data_cuda_ssd/minsu/109_cj_optical_flow/fold2/test/"
SAVE_DESCRIPT="histo_fold2_per_1flow_109"
SAVE_DIR="/data_cuda_ssd/minsu/0419_new_cj_binary_cross_validation/${SAVE_DESCRIPT}"
python main_histogram_svm.py --lr=0.00002 --aug=1.0 --epoch=50 --batch_size=20 --batch_size_test=1 --image_T=1 --optim=sgd --mode=2 --reverse_axis="" \
                             --cpus=16 --gpu=0 --resume=True --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --clip_frames=1 \
                             --model=vivit_2 --description="Binary | Fold 2 | 109 | 20 batch | 1 clip" --flow=True --test

DATA_DIR="/data_cuda_ssd/minsu/109_cj_optical_flow/fold3/test/"
SAVE_DESCRIPT="histo_fold3_per_1flow_109"
SAVE_DIR="/data_cuda_ssd/minsu/0419_new_cj_binary_cross_validation/${SAVE_DESCRIPT}"
python main_histogram_svm.py --lr=0.00002 --aug=1.0 --epoch=50 --batch_size=20 --batch_size_test=1 --image_T=1 --optim=sgd --mode=2 --reverse_axis="" \
                             --cpus=16 --gpu=0 --resume=True --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --clip_frames=1 \
                             --model=vivit_2 --description="Binary | Fold 3 | 109 | 20 batch | 1 clip" --flow=True --test

DATA_DIR="/data_cuda_ssd/minsu/109_cj_optical_flow/fold4/test/"
SAVE_DESCRIPT="histo_fold4_per_1flow_109"
SAVE_DIR="/data_cuda_ssd/minsu/0419_new_cj_binary_cross_validation/${SAVE_DESCRIPT}"
python main_histogram_svm.py --lr=0.00002 --aug=1.0 --epoch=50 --batch_size=20 --batch_size_test=1 --image_T=1 --optim=sgd --mode=2 --reverse_axis="" \
                             --cpus=16 --gpu=0 --resume=True --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --clip_frames=1 \
                             --model=vivit_2 --description="Binary | Fold 4 | 109 | 20 batch | 1 clip" --flow=True --test

DATA_DIR="/data_cuda_ssd/minsu/109_cj_optical_flow/fold5/test/"
SAVE_DESCRIPT="histo_fold5_per_1flow_109"
SAVE_DIR="/data_cuda_ssd/minsu/0419_new_cj_binary_cross_validation/${SAVE_DESCRIPT}"
python main_histogram_svm.py --lr=0.00002 --aug=1.0 --epoch=50 --batch_size=20 --batch_size_test=1 --image_T=1 --optim=sgd --mode=2 --reverse_axis="" \
                             --cpus=16 --gpu=0 --resume=True --data_dir=${DATA_DIR} --save_dir="${SAVE_DIR}" --clip_frames=1 \
                             --model=vivit_2 --description="Binary | Fold 5 | 109 | 20 batch | 1 clip" --flow=True --test

