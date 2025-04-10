# [CVPR'2025] - SGLATrack

The official implementation for the **CVPR 2025** paper

\[[_Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking_](https://arxiv.org/abs/2503.06625)\]

[[Models](https://drive.google.com/drive/folders/1sHL7aFVZFwkPy6js48x-EKfoZC7oJc9X?usp=sharing)], [[Raw Results](https://drive.google.com/drive/folders/1ss-KQqPsfIXeOcl_h3w6Q09dEk07DjUy?usp=sharing)]


### :star2: Performance on Aerial Datasets

| Tracker      | UAV123 (AUC) | UAV123_10FPS (AUC) | UAVDT (AUC) | DTB70 (AUC) | UAVTrack112 (AUC) | UAVTrack_L (AUC) |
|:------------:|:------------:|:-----------:|:-----------------:|:---------------:|:------------:|:-----------:|
| SGLATrack-DeiT* | 66.9       | 65.5       | 59.9              | 65.1            | 67.5         | 64.0     |
| SGLATrack-ViT | 66.1         | 64.5       | 60.0              | 65.8            | 67.3         | 64.3       | 
| SGLATrack-EVA | 65.1         | 64.3        | 57.9            | 63.8           | 66.9         | 64.7        | 

### :star2: Performance on Generic Datasets
| Tracker      | TrackingNet (AUC) | LaSOT (AUC) | GOT-10k (AO) |
|:------------:|:------------:|:-----------:|:-----------------:|
| SGLATrack-DeiT* | 79.5      | 63.0       | 66.3             |
| SGLATrack-ViT | 79.4        | 64.1       | 66.0             |
| SGLATrack-EVA | 77.7         | 60.9       | 64.2            |


## Training Data Preparation
Put the training datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```

## Test Data Preparation
For ease of testing, we have made the structured dataset available for download at [here](https://pan.baidu.com/s/1MaeGLRcAUbJxksbF_CrOeQ?pwd=5vbv), code: 5vbv.
Put the test datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- UAV123
            |-- anno
            |-- data_seq
        -- UAV123_10fps
            |-- anno
            |-- data_seq
        -- uavdt
            |-- anno
            |-- sequences
        -- V4RFlight112
            |-- anno
            |-- anno_l
            |-- data_seq
            |-- attributes
        -- DTB70
            |-- Animal1
            |-- Animal2
            ...
        -- VisDrone2018-SOT-test-dev
            |-- annotations
            |-- sequences
            |-- attributes
   ```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```


## Training
Download pre-trained [DeiT-tiny distilled weights](https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth) and put it under `$PROJECT_ROOT$/pretrained_models` 

```
python tracking/train.py \
--script sglatrack --config deit_distilled \
--save_dir ./output \
--mode multiple --nproc_per_node 4 \
--use_wandb 0
```

Replace `--config` with the desired model config under `experiments/sglatrack`.

We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Test and Evaluation
- UAV123 or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py --tracker_param sglatrack --dataset uav123 --threads 8 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- uav123_10fps
```
python tracking/test.py  --tracker_param sglatrack --dataset uav123_10fps --threads 8 --num_gpus 4
```
- uavtrack_L
```
python tracking/test.py  --tracker_param sglatrack --dataset uavtrack --threads 8 --num_gpus 4
```
- uavtrack112
```
python tracking/test.py  --tracker_param sglatrack --dataset uavtrack112 --threads 8 --num_gpus 4
```
- uavdt
```
python tracking/test.py  --tracker_param sglatrack --dataset uavdt --threads 8 --num_gpus 4
```
- dtb70
```
python tracking/test.py  --tracker_param sglatrack --dataset dtb70 --threads 8 --num_gpus 4
```
- visdrone
```
python tracking/test.py  --tracker_param sglatrack --dataset visdrone --threads 8 --num_gpus 4
```

## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX2080Ti GPU.

```
python tracking/profile_model.py
```


## Contact
For any questions or cooperation, please contact xcc23cg@163.com


## Acknowledgments
* Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) and [AVTrack](https://github.com/wuyou3474/AVTrack) library, which helps us to quickly implement our ideas.



## Citation
If our work is useful for your research, please consider citing:

```Bibtex
@inproceedings{sglatrack,
  title={Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking},
  author={Xue, Chaocan and Zhong, Bineng and Liang, Qihua and Zheng, Yaozong and Li, Ning and Xue, Yuanliang and Song, Shuxiang},
  booktitle={CVPR},
  year={2025}
}
```
