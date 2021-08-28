# Prepare ScanNet Data
- Download origin [ScanNet](https://github.com/ScanNet/ScanNet) v2 data
```sh
dataset
└── scannetv2
    ├── meta_data(unnecessary, we have moved into our source code)
    │   ├── scannetv2_train.txt
    │   ├── scannetv2_val.txt
    │   ├── scannetv2_test.txt
    │   └── scannetv2-labels.combined.tsv
    ├── scans
    │   ├── ...
    │   ├── [scene_id]
    |   |    └── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
    |   └── ...
    └── scans_test
        ├── ...
        ├── [scene_id]
        |    └── [scene_id]_vh_clean_2.ply & [scene_id].txt
        └── ...
```

- Refer to [PointGroup](https://github.com/Jia-Research-Lab/PointGroup), we've modify the code, and it can generate input files `[scene_id]_inst_nostuff.pth` for instance segmentation directly, you don't need to split the origin data into `train/val/test`, the script refer to `gorilla3d/preprocessing/scannetv2/segmentation`.
- And we package these command. You just running:
```sh
sh prepare_data.sh
```
- After running such command, the structure of directory is as following:
```sh
dataset
└── scannetv2
    ├── meta_data(unnecessary, we have moved into our source code)
    │   └── ...
    ├── scans
    |   └── ...
    ├── scans_test
    |   └── ...
    |   (data preparation generation as following)
    ├── train
    |   ├── [scene_id]_inst_nostuff.pth
    |   └── ...
    ├── test
    |   ├── [scene_id]_inst_nostuff.pth
    |   └── ...
    ├── val
    |   ├── [scene_id]_inst_nostuff.pth
    |   └── ...
    └── val_gt
        ├── [scene_id].txt
        └── ...
```
