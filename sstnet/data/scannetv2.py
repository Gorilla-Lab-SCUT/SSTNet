# Copyright (c) Gorilla-Lab. All rights reserved.
import math
import glob
import os.path as osp
from typing import Dict, List, Sequence, Tuple, Union

import gorilla
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset

import segmentator
import pointgroup_ops
from .utils import elastic

@gorilla.DATASETS.register_module(force=True)
class ScanNetV2Inst(Dataset):
    def __init__(self,
                 data_root: str,
                 full_scale: List[int]=[128, 512],
                 scale: int=50,
                 max_npoint: int=250000,
                 task: str="train",
                 with_elastic: bool=False,
                 test_mode: bool=False,
                 with_superpoints: bool=True,
                 **kwargs):
        # initialize dataset parameters
        self.logger = gorilla.derive_logger(__name__)
        self.data_root = data_root
        self.full_scale = full_scale
        self.scale = scale
        self.max_npoint = max_npoint
        self.test_mode = test_mode
        self.with_elastic = with_elastic
        self.with_superpoints = with_superpoints
        self.task = task
        self.aug_flag = "train" in self.task
        
        # load files
        self.load_files()
    
    def load_files(self):
        file_names = sorted(glob.glob(osp.join(self.data_root, self.task, "*.pth")))
        self.files = [torch.load(i) for i in gorilla.track(file_names)]
        self.logger.info(f"{self.task} samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple:
        if "test" in self.task:
            sub_dir = "scans_test"
            xyz_origin, rgb, faces, scene = self.files[index]
            # construct fake label for label-lack testset
            semantic_label = np.zeros(xyz_origin.shape[0], dtype=np.int32)
            instance_label = np.zeros(xyz_origin.shape[0], dtype=np.int32)
        else:
            sub_dir = "scans"
            xyz_origin, rgb, faces, semantic_label, instance_label, coords_shift, scene = self.files[index]

        mesh_file = osp.join(osp.join(self.data_root, sub_dir, scene, scene+"_vh_clean_2.ply"))
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
        faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
        superpoint = segmentator.segment_mesh(vertices, faces).numpy()

        ### jitter / flip x / rotation
        if self.aug_flag:
            xyz_middle = self.data_aug(xyz_origin, True, True, True)
        else:
            xyz_middle = self.data_aug(xyz_origin, False, False, False)

        ### scale
        xyz = xyz_middle * self.scale

        ### elastic
        if self.with_elastic:
            xyz = elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

        ### offset
        xyz_offset = xyz.min(0)
        xyz -= xyz_offset

        ### crop
        valid_idxs = np.ones(len(xyz_middle), dtype=np.bool)
        if not self.test_mode:
            xyz, valid_idxs = self.crop(xyz)

        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)

        ### get instance information
        inst_num, inst_infos = self.get_instance_info(xyz_middle, instance_label.astype(np.int32))
        inst_info = inst_infos["instance_info"]  # [n, 9], (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        inst_pointnum = inst_infos["instance_pointnum"]   # [num_inst], list
        
        loc = torch.from_numpy(xyz).long()
        loc_offset = torch.from_numpy(xyz_offset).long()
        loc_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb)
        if self.aug_flag:
            feat += torch.randn(3) * 0.1
        semantic_label = torch.from_numpy(semantic_label)
        instance_label = torch.from_numpy(instance_label)
        superpoint = torch.from_numpy(superpoint)

        inst_info = torch.from_numpy(inst_info)

        return scene, loc, loc_offset, loc_float, feat, semantic_label, instance_label, superpoint, inst_num, inst_info, inst_pointnum

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)

    def crop(self, xyz: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        r"""
        crop the point cloud to reduce training complexity

        Args:
            xyz (np.ndarray, [N, 3]): input point cloud to be cropped

        Returns:
            Union[np.ndarray, np.ndarray]: processed point cloud and boolean valid indices
        """
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def get_instance_info(self,
                          xyz: np.ndarray,
                          instance_label: np.ndarray) -> Union[int, Dict]:
        r"""
        get the informations of instances (amount and coordinates)

        Args:
            xyz (np.ndarray, [N, 3]): input point cloud data
            instance_label (np.ndarray, [N]): instance ids of point cloud

        Returns:
            Union[int, Dict]: the amount of instances andinformations
                              (coordinates and the number of points) of instances
        """
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # [n, 9], float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # [num_inst], int
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}

    def get_cropped_inst_label(self,
                               instance_label: np.ndarray,
                               valid_idxs: np.ndarray) -> np.ndarray:
        r"""
        get the instance labels after crop operation and recompact

        Args:
            instance_label (np.ndarray, [N]): instance label ids of point cloud
            valid_idxs (np.ndarray, [N]): boolean valid indices

        Returns:
            np.ndarray: processed instance labels
        """
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
        locs = []
        loc_offset_list = []
        locs_float = []
        feats = []
        semantic_labels = []
        instance_labels = []

        instance_infos = []  # [N, 9]
        instance_pointnum = []  # [total_num_inst], int

        batch_offsets = [0]
        scene_list = []
        superpoint_list = []
        superpoint_bias = 0

        total_inst_num = 0
        for i, data in enumerate(batch):
            scene, loc, loc_offset, loc_float, feat, semantic_label, instance_label, superpoint, inst_num, inst_info, inst_pointnum = data
            
            scene_list.append(scene)
            superpoint += superpoint_bias
            superpoint_bias += (superpoint.max() + 1)

            invalid_ids = np.where(instance_label != -100)
            instance_label[invalid_ids] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + loc.shape[0])

            locs.append(torch.cat([torch.LongTensor(loc.shape[0], 1).fill_(i), loc], 1))
            loc_offset_list.append(loc_offset)
            locs_float.append(loc_float)
            feats.append(feat)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
            superpoint_list.append(superpoint)

            instance_infos.append(inst_info)
            instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]

        locs = torch.cat(locs, 0)                                # long [N, 1 + 3], the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float [N, 3]
        superpoint = torch.cat(superpoint_list, 0).long()               # long[N]
        feats = torch.cat(feats, 0)                              # float [N, C]
        semantic_labels = torch.cat(semantic_labels, 0).long()                     # long [N]
        instance_labels = torch.cat(instance_labels, 0).long()   # long [N]
        locs_offset = torch.stack(loc_offset_list)               # long [B, 3]

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float [N, 9] (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int [total_num_inst]

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None) # long [3]

        ### voxelize
        batch_size = len(batch)
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, batch_size, 4)

        return {"locs": locs, "locs_offset": locs_offset, "voxel_locs": voxel_locs,
                "scene_list": scene_list, "p2v_map": p2v_map, "v2p_map": v2p_map,
                "locs_float": locs_float, "feats": feats,
                "semantic_labels": semantic_labels, "instance_labels": instance_labels,
                "instance_info": instance_infos, "instance_pointnum": instance_pointnum,
                "offsets": batch_offsets, "spatial_shape": spatial_shape, "superpoint": superpoint}

