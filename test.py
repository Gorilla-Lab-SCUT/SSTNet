# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse
import numpy as np
import os

import torch
import spconv
import scipy.stats as stats

import pointgroup_ops
import gorilla
import gorilla3d
import sstnet

def get_parser():
    parser = argparse.ArgumentParser(description="SSTNet for Point Cloud Instance Segmentation")
    parser.add_argument("--config",
                        type=str,
                        default="config/default.yaml",
                        help="path to config file")
    ### pretrain
    parser.add_argument("--pretrain",
                        type=str,
                        default="",
                        help="path to pretrain model")
    ### split
    parser.add_argument("--split",
                        type=str,
                        default="val",
                        help="dataset split to test")
    ### semantic only
    parser.add_argument("--semantic",
                        action="store_true",
                        help="only evaluate semantic segmentation")
    ### log file path
    parser.add_argument("--log-file",
                        type=str,
                        default=None,
                        help="log_file path")
    ### test srcipt operation
    parser.add_argument("--eval",
                        action="store_true",
                        help="evaluate or not")
    parser.add_argument("--save",
                        action="store_true",
                        help="save results or not")
    parser.add_argument("--visual",
                        type=str,
                        default=None,
                        help="visual path, give to save visualization results")

    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    cfg = gorilla.Config.fromfile(args.config)
    cfg.pretrain = args.pretrain
    cfg.semantic = args.semantic
    cfg.dataset.task = args.split  # change tasks
    cfg.data.visual = args.visual
    cfg.data.eval = args.eval
    cfg.data.save = args.save

    gorilla.set_random_seed(cfg.data.test_seed)

    #### get logger file
    params_dict = dict(
        epoch=cfg.data.test_epoch,
        optim=cfg.optimizer.type,
        lr=cfg.optimizer.lr,
        scheduler=cfg.lr_scheduler.type
    )
    if "test" in args.split:
        params_dict["suffix"] = "test"

    log_dir, logger = gorilla.collect_logger(
        prefix=os.path.splitext(args.config.split("/")[-1])[0],
        log_name="test",
        log_file=args.log_file,
        # **params_dict
    )

    logger.info(
        "************************ Start Logging ************************")

    # log the config
    logger.info(cfg)

    global result_dir
    result_dir = os.path.join(
        log_dir, "result",
        "epoch{}_nmst{}_scoret{}_npointt{}".format(cfg.data.test_epoch,
                                                   cfg.data.TEST_NMS_THRESH,
                                                   cfg.data.TEST_SCORE_THRESH,
                                                   cfg.data.TEST_NPOINT_THRESH),
        args.split)
    os.makedirs(os.path.join(result_dir, "predicted_masks"), exist_ok=True)

    global semantic_label_idx
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    semantic_label_idx = torch.tensor([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
    ]).cuda()

    return logger, cfg


def test(model, cfg, logger):
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

    epoch = cfg.data.test_epoch
    semantic = cfg.semantic

    cfg.dataset.test_mode = True
    cfg.dataloader.batch_size = 1
    cfg.dataloader.num_workers = 2
    test_dataset = gorilla.build_dataset(cfg.dataset)
    test_dataloader = gorilla.build_dataloader(test_dataset, cfg.dataloader)

    with torch.no_grad():
        model = model.eval()

        # init timer to calculate time
        timer = gorilla.Timer()

        # define evaluator
        # get the real data root
        data_root = os.path.join(os.path.dirname(__file__), cfg.dataset.data_root)
        sub_dir = "scans_test" if "test" in cfg.dataset.task else "scans"

        semantic_dataset_root = os.path.join(data_root, "scans")
        instance_dataset_root = os.path.join(data_root, cfg.dataset.task + "_gt")
        evaluator = gorilla3d.ScanNetSemanticEvaluator(semantic_dataset_root)
        inst_evaluator = gorilla3d.ScanNetInstanceEvaluator(instance_dataset_root)

        for i, batch in enumerate(test_dataloader):
            torch.cuda.empty_cache()
            timer.reset()
            N = batch["feats"].shape[0]
            test_scene_name = batch["scene_list"][0]

            coords = batch["locs"].cuda() # [N, 1 + 3] dimension 0 for batch_idx
            locs_offset = batch["locs_offset"].cuda() # [B, 3]
            voxel_coords = batch["voxel_locs"].cuda() # [M, 1 + 3]
            p2v_map = batch["p2v_map"].cuda() # [N]
            v2p_map = batch["v2p_map"].cuda() # [M, 1 + maxActive]

            coords_float = batch["locs_float"].cuda() # [N, 3]
            feats = batch["feats"].cuda() # [N, C]

            batch_offsets = batch["offsets"].cuda() # [B + 1]
            scene_list = batch["scene_list"]
            superpoint = batch["superpoint"].cuda() # [N
            _, superpoint = torch.unique(superpoint, return_inverse=True)  # [N]

            extra_data = {"batch_idxs": coords[:, 0].int(),
                          "superpoint": superpoint,
                          "locs_offset": locs_offset,
                          "scene_list": scene_list}

            spatial_shape = batch["spatial_shape"]

            if cfg.model.use_coords:
                feats = torch.cat((feats, coords_float), 1)
            voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.data.mode) # [M, C]

            input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.dataloader.batch_size)

            data_time = timer.since_last()

            ret = model(input_,
                        p2v_map,
                        feats,
                        coords_float,
                        epoch,
                        extra_data,
                        mode="test",
                        semantic_only=semantic)

            semantic_scores = ret["semantic_scores"] # [N, nClass]
            pt_offsets = ret["pt_offsets"] # [N, 3]

            score_epochs = cfg.model.score_epochs
            prepare_flag = epoch > score_epochs
            if prepare_flag and not semantic:
                scores = ret["proposal_scores"]

            ##### preds
            with torch.no_grad():
                preds = {}
                preds["semantic"] = semantic_scores
                preds["pt_offsets"] = pt_offsets
                if prepare_flag and not semantic:
                    proposals_idx, proposals_offset = ret["proposals"]
                    preds["score"] = scores
                    preds["proposals"] = (proposals_idx, proposals_offset)


            ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
            semantic_scores = preds["semantic"]  # [N, nClass=20]
            semantic_pred = semantic_scores.max(1)[1]  # [N]

            pt_offsets = preds["pt_offsets"]  # [N, 3]

            ##### semantic segmentation evaluation
            if cfg.data.eval:
                inputs = [{"scene_name": test_scene_name}]
                outputs = [{"semantic_pred": semantic_pred}]
                evaluator.process(inputs, outputs)

            if prepare_flag and not semantic:
                scores = preds["score"]  # [num_prop, 1]
                scores_pred = torch.sigmoid(scores.view(-1))

                proposals_idx, proposals_offset = preds["proposals"]
                # proposals_idx: (sumNPoint, 2) dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (num_prop + 1)
                proposals_pred = torch.zeros(
                    (proposals_offset.shape[0] - 1, N),
                    dtype=torch.int,
                    device=scores_pred.device)  # [num_prop, N]
                proposals_pred[proposals_idx[:, 0].long(),
                               proposals_idx[:, 1].long()] = 1
                semantic_pred_list = []
                for start, end in zip(proposals_offset[:-1],
                                      proposals_offset[1:]):
                    semantic_label, _ = stats.mode(
                        semantic_pred[proposals_idx[start:end,
                                                    1].long()].cpu().numpy())
                    semantic_label = semantic_label[0]
                    semantic_pred_list.append(semantic_label)

                semantic_id = semantic_label_idx[semantic_pred_list]

                ##### score threshold
                score_mask = (scores_pred > cfg.data.TEST_SCORE_THRESH)
                scores_pred = scores_pred[score_mask]
                proposals_pred = proposals_pred[score_mask]
                semantic_id = semantic_id[score_mask]

                ##### npoint threshold
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum > cfg.data.TEST_NPOINT_THRESH)
                scores_pred = scores_pred[npoint_mask]
                proposals_pred = proposals_pred[npoint_mask]
                semantic_id = semantic_id[npoint_mask]

                ##### nms
                if semantic_id.shape[0] == 0:
                    pick_idxs = np.empty(0)
                else:
                    proposals_pred_f = proposals_pred.float(
                    )  # [num_prop, N], float, cuda
                    intersection = torch.mm(
                        proposals_pred_f, proposals_pred_f.t(
                        ))  # [num_prop, num_prop], float, cuda
                    proposals_pointnum = proposals_pred_f.sum(
                        1)  # [num_prop], float, cuda
                    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(
                        1, proposals_pointnum.shape[0])
                    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(
                        proposals_pointnum.shape[0], 1)
                    cross_ious = intersection / (proposals_pn_h +
                                                 proposals_pn_v - intersection)
                                                 
                    pick_idxs = gorilla3d.non_max_suppression(
                        cross_ious.cpu().numpy(),
                        scores_pred.cpu().numpy(),
                        cfg.data.TEST_NMS_THRESH)  # int, (nCluster, N)
                clusters = proposals_pred[pick_idxs]
                cluster_scores = scores_pred[pick_idxs]
                cluster_semantic_id = semantic_id[pick_idxs]

                nclusters = clusters.shape[0]

                ##### prepare for evaluation
                if cfg.data.eval:
                    pred_info = {}
                    pred_info["scene_name"] = test_scene_name
                    pred_info["conf"] = cluster_scores.cpu().numpy()
                    pred_info["label_id"] = cluster_semantic_id.cpu().numpy()
                    pred_info["mask"] = clusters.cpu().numpy()
                    inst_evaluator.process(inputs, [pred_info])

            inference_time = timer.since_last()

            ##### visual
            if cfg.data.visual is not None:
                # visual semantic result
                gorilla.check_dir(cfg.data.visual)
                if cfg.semantic:
                    pass
                # visual instance result
                else:
                    gorilla3d.visualize_instance_mask(clusters.cpu().numpy(),
                                                      test_scene_name,
                                                      cfg.data.visual,
                                                      os.path.join(data_root, sub_dir),
                                                      cluster_scores.cpu().numpy(),
                                                      semantic_pred.cpu().numpy(),)

            ##### save files
            if (prepare_flag and cfg.data.save):
                f = open(os.path.join(result_dir, test_scene_name + ".txt"), "w")
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()  # [N]
                    semantic_label = np.argmax(
                        np.bincount(
                            semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                    score = cluster_scores[proposal_id]
                    f.write(f"predicted_masks/{test_scene_name}_{proposal_id:03d}.txt "
                            f"{semantic_label_idx[semantic_label]} {score:.4f}")
                    if proposal_id < nclusters - 1:
                        f.write("\n")
                    content = list(map(lambda x: str(x), clusters_i.tolist()))
                    content = "\n".join(content)
                    with open(
                            os.path.join(
                                result_dir, "predicted_masks",
                                test_scene_name + "_%03d.txt" % (proposal_id)),
                            "w") as cf:
                        cf.write(content)
                    # np.savetxt(os.path.join(result_dir, "predicted_masks", test_scene_name + "_%03d.txt" % (proposal_id)), clusters_i, fmt="%d")
                f.close()

            save_time = timer.since_last()
            total_time = timer.since_start()

            ##### print
            if semantic:
                logger.info(
                    f"instance iter: {i + 1}/{len(test_dataloader)} point_num: {N} "
                    f"time: total {total_time:.2f}s data: {data_time:.2f}s "
                    f"inference {inference_time:.2f}s save {save_time:.2f}s")
            else:
                logger.info(
                    f"instance iter: {i + 1}/{len(test_dataloader)} point_num: {N} "
                    f"ncluster: {nclusters} time: total {total_time:.2f}s data: {data_time:.2f}s "
                    f"inference {inference_time:.2f}s save {save_time:.2f}s")

        ##### evaluation
        if cfg.data.eval:
            if not semantic:
                inst_evaluator.evaluate(prec_rec=False)
            evaluator.evaluate()


if __name__ == "__main__":
    logger, cfg = init()

    ##### model
    logger.info("=> creating model ...")
    logger.info(f"Classes: {cfg.model.classes}")

    model = gorilla.build_model(cfg.model)

    use_cuda = torch.cuda.is_available()
    logger.info(f"cuda available: {use_cuda}")
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info(f"#classifier parameters (model): {sum([x.nelement() for x in model.parameters()])}")

    ##### load model
    gorilla.load_checkpoint(
        model, cfg.pretrain
    )  # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, cfg, logger)
