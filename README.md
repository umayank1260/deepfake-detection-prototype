# Deepfake Detection Prototype
## This model using the Dataset available on kaggle known as deepfake detection challenge.
## Solution Description :-
### In general solution is based on frame-by-frame classification approach. Other complex things did not work so well on public leaderboard.

Face-Detector
MTCNN detector is chosen due to kernel time limits. It would be better to use S3FD detector as more precise and robust, but opensource Pytorch implementations don't have a license.

Input size for face detector was calculated for each video depending on video resolution.

2x scale for videos with less than 300 pixels wider side
no rescale for videos with wider side between 300 and 1000
0.5x scale for videos with wider side > 1000 pixels
0.33x scale for videos with wider side > 1900 pixels
Input size
As soon as I discovered that EfficientNets significantly outperform other encoders I used only them in my solution. As I started with B4 I decided to use "native" size for that network (380x380). Due to memory costraints I did not increase input size even for B7 encoder.

Margin
When I generated crops for training I added 30% of face crop size from each side and used only this setting during the competition. See extract_crops.py for the details

Encoders
The winning encoder is current state-of-the-art model (EfficientNet B7) pretrained with ImageNet and noisy student Self-training with Noisy Student improves ImageNet classification

Averaging predictions
I used 32 frames for each video. For each model output instead of simple averaging I used the following heuristic which worked quite well on public leaderbord (0.25 -> 0.22 solo B5).
### Augmentations
I used heavy augmentations by default. Albumentations library supports most of the augmentations out of the box. Only needed to add IsotropicResize augmentation.
Building docker image
All libraries and enviroment is already configured with Dockerfile. It requires docker engine https://docs.docker.com/engine/install/ubuntu/ and nvidia docker in your system https://github.com/NVIDIA/nvidia-docker.

To build a docker image run docker build -t df .

Running docker
docker run --runtime=nvidia --ipc=host --rm  --volume <DATA_ROOT>:/dataset -it df

Data preparation
Once DFDC dataset is downloaded all the scripts expect to have dfdc_train_xxx folders under data root directory.

Preprocessing is done in a single script preprocess_data.sh which requires dataset directory as first argument. It will execute the steps below:

1. Find face bboxes
To extract face bboxes I used facenet library, basically only MTCNN. python preprocessing/detect_original_faces.py --root-dir DATA_ROOT This script will detect faces in real videos and store them as jsons in DATA_ROOT/bboxes directory

2. Extract crops from videos
To extract image crops I used bboxes saved before. It will use bounding boxes from original videos for face videos as well. python preprocessing/extract_crops.py --root-dir DATA_ROOT --crops-dir crops This script will extract face crops from videos and save them in DATA_ROOT/crops directory

3. Generate landmarks
From the saved crops it is quite fast to process crops with MTCNN and extract landmarks
python preprocessing/generate_landmarks.py --root-dir DATA_ROOT This script will extract landmarks and save them in DATA_ROOT/landmarks directory

4. Generate diff SSIM masks
python preprocessing/generate_diffs.py --root-dir DATA_ROOT This script will extract SSIM difference masks between real and fake images and save them in DATA_ROOT/diffs directory

5. Generate folds
python preprocessing/generate_folds.py --root-dir DATA_ROOT --out folds.csv By default it will use 16 splits to have 0-2 folders as a holdout set. Though only 400 videos can be used for validation as well.

Training
Training 5 B7 models with different seeds is done in train.sh script.

During training checkpoints are saved for every epoch.

Hardware requirements
Mostly trained on devbox configuration with 4xTitan V, thanks to Nvidia and DSB2018 competition where I got these gpus https://www.kaggle.com/c/data-science-bowl-2018/

Overall training requires 4 GPUs with 12gb+ memory. Batch size needs to be adjusted for standard 1080Ti or 2080Ti graphic cards.

As I computed fake loss and real loss separately inside each batch, results might be better with larger batch size, for example on V100 gpus. Even though SyncBN is used larger batch on each GPU will lead to less noise as DFDC dataset has some fakes where face detector failed and face crops are not really fakes.
