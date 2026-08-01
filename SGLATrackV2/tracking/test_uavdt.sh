#!/bin/bash  
# python  tracking/train.py --script mfi_track --config vitb_256_mfit_32x1_1e4_lasher_15ep_sot_rgbt --save_dir ./output/vitb_256_mfit_32x1_1e4_lasher_15ep_sot_rgbt --mode multiple --nproc_per_node 2


for ((runid=300;runid>294;runid--))
do
    python tracking/test.py sglatrackv2 baseline_WOCE_cae --dataset_name visdrone --threads 10 --num_gpus 2 --runid $runid  && echo "Command for runid $runid executed successfully"
    python tracking/test.py sglatrackv2 baseline_WOCE_cae --dataset_name uavdt --threads 10 --num_gpus 2 --runid $runid  && echo "Command for runid $runid executed successfully"
done




