# Copyright (c) Gorilla-Lab. All rights reserved.
import glob
import os.path as osp

import torch
import gorilla
import gorilla3d
import spconv

import sstnet
import pointgroup_ops

def get_parser():
    # the default argument parser contains some 
    # essential parameters for distributed
    parser = gorilla.core.default_argument_parser()
    parser.add_argument("--config",
                        type=str,
                        default="config/default.yaml",
                        help="path to config file")

    args_cfg = parser.parse_args()

    return args_cfg


def do_train(model, cfg, logger):
    model.train()
    # initilize optimizer and scheduler (scheduler is optional-adjust learning rate manually)
    optimizer = gorilla.build_optimizer(model, cfg.optimizer)
    lr_scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)

    # initialize criterion (Optional, can calculate in model forward)
    criterion = gorilla.build_loss(cfg.loss)
    
    # resume model/optimizer/scheduler
    iter = 1
    checkpoint, epoch = get_checkpoint(cfg.log_dir)
    if gorilla.is_filepath(checkpoint): # read valid checkpoint file
        # meta is the dict save some necessary information (last epoch/iteration, acc, loss)
        meta = gorilla.resume(model=model,
                              filename=checkpoint,
                              optimizer=optimizer,     # optimizer and scheduler is optional
                              scheduler=lr_scheduler,  # to resume (can not give these paramters)
                              resume_optimizer=True,
                              resume_scheduler=True,
                              strict=False,
                              )
        # get epoch from meta (Optional)
        epoch = meta.get("epoch", epoch) + 1
        iter = meta.get("iter", iter) + 1
    
    # initialize train dataset
    train_dataset = gorilla.build_dataset(cfg.dataset)
    train_dataloader = gorilla.build_dataloader(train_dataset,
                                                cfg.dataloader,
                                                shuffle=True,
                                                pin_memory=True,
                                                drop_last=True)

    # initialize tensorboard (Optional) TODO: integrating the tensorborad manager
    writer = gorilla.TensorBoardWriter(cfg.log_dir) # tensorboard writer

    # initialize timers (Optional)
    iter_timer = gorilla.Timer()
    epoch_timer = gorilla.Timer()

    # loss/time buffer for epoch record (Optional)
    loss_buffer = gorilla.HistoryBuffer()
    iter_time = gorilla.HistoryBuffer()
    data_time = gorilla.HistoryBuffer()

    while epoch <= cfg.data.epochs:
        for i, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache() # (empty cuda cache, Optional)
            # calculate data loading time
            data_time.update(iter_timer.since_last())
            # cuda manually (TODO: integrating the data cuda operation)
            ##### prepare input and forward
            coords = batch["locs"].cuda() # [N, 1 + 3], long, cuda, dimension 0 for batch_idx
            locs_offset = batch["locs_offset"].cuda()  # [B, 3], long, cuda
            voxel_coords = batch["voxel_locs"].cuda()  # [M, 1 + 3], long, cuda
            p2v_map = batch["p2v_map"].cuda()  # [N], int, cuda
            v2p_map = batch["v2p_map"].cuda()  # [M, 1 + maxActive], int, cuda

            coords_float = batch["locs_float"].cuda()  # [N, 3], float32, cuda
            feats = batch["feats"].cuda()  # [N, C], float32, cuda
            semantic_labels = batch["semantic_labels"].cuda()  # [N], long, cuda
            instance_labels = batch["instance_labels"].cuda(
            )  # [N], long, cuda, 0~total_num_inst, -100

            instance_info = batch["instance_info"].cuda(
            )  # [N, 9], float32, cuda, (meanxyz, minxyz, maxxyz)
            instance_pointnum = batch["instance_pointnum"].cuda(
            )  # [total_num_inst], int, cuda

            batch_offsets = batch["offsets"].cuda()  # [B + 1], int, cuda
            superpoint = batch["superpoint"].cuda()  # [N], long, cuda
            _, superpoint = torch.unique(superpoint, return_inverse=True)  # [N], long, cuda

            fusion_epochs = cfg.model.fusion_epochs
            score_epochs = cfg.model.score_epochs
            prepare_flag = (epoch > score_epochs)
            fusion_flag = (epoch > fusion_epochs)
            with_refine = cfg.model.with_refine
            scene_list = batch["scene_list"]
            spatial_shape = batch["spatial_shape"]

            extra_data = {
                "batch_idxs": coords[:, 0].int(),
                "superpoint": superpoint,
                "locs_offset": locs_offset,
                "scene_list": scene_list,
                "instance_labels": instance_labels,
                "instance_pointnum": instance_pointnum
            }

            if cfg.model.use_coords:
                feats = torch.cat((feats, coords_float), 1)
            voxel_feats = pointgroup_ops.voxelization(
                feats, v2p_map, cfg.data.mode)  # [M, C]

            input_ = spconv.SparseConvTensor(voxel_feats,
                                             voxel_coords.int(),
                                             spatial_shape,
                                             cfg.dataloader.batch_size)

            ret = model(input_,
                        p2v_map,
                        coords_float,
                        epoch,
                        extra_data)

            semantic_scores = ret["semantic_scores"]  # [N, nClass] float32, cuda
            pt_offsets = ret["pt_offsets"]  # [N, 3], float32, cuda

            loss_inp = {}
            loss_inp["batch_idxs"] = coords[:, 0].int()
            loss_inp["feats"] = feats
            loss_inp["scene_list"] = scene_list

            loss_inp["semantic_scores"] = (semantic_scores, semantic_labels)
            loss_inp["pt_offsets"] = (pt_offsets,
                                      coords_float,
                                      instance_info,
                                      instance_labels,
                                      instance_pointnum)

            loss_inp["superpoint"] = superpoint
            loss_inp["empty_flag"] = ret["empty_flag"] # avoid stack error

            if fusion_flag:
                loss_inp["fusion"] = ret["fusion"]

            if with_refine:
                loss_inp["refine"] = ret["refine"]

            if prepare_flag:
                loss_inp["proposals"] = ret["proposals"]
                scores = ret["proposal_scores"]
                # scores: (num_prop, 1) float, cuda
                # proposals_idx: (sum_points, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (num_prop + 1), int, cpu

                loss_inp["proposal_scores"] = scores

            loss, loss_out = criterion(loss_inp, epoch)
            loss_buffer.update(loss)

            # sample the learning rate(Optional)
            lr = optimizer.param_groups[0]["lr"]
            # write tensorboard
            loss_out.update({"loss": loss, "lr": lr})
            writer.update(loss_out, iter)
            # # equivalent write operation
            # writer.add_scalar(f"train/loss", loss, iter)
            # writer.add_scalar(f"lr", lr, iter)
            # # (NOTE: the `loss_out` is work for multi losses, which saves each loss item)
            # for k, v in loss_out.items():
            #     writer.add_scalar(f"train/{k}", v[0], iter)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter += 1

            # calculate time and reset timer(Optional)
            iter_time.update(iter_timer.since_start())
            iter_timer.reset() # record the iteration time and reset timer

            # TODO: the time manager will be integrated into gorilla-core
            # calculate remain time(Optional)
            remain_iter = (cfg.data.epochs - epoch + 1) * len(train_dataloader) + i + 1
            remain_time = gorilla.convert_seconds(remain_iter * iter_time.avg) # convert seconds into "hours:minutes:sceonds"

            print(f"epoch: {epoch}/{cfg.data.epochs} iter: {i + 1}/{len(train_dataloader)} "
                  f"lr: {lr:4f} loss: {loss_buffer.latest:.4f}({loss_buffer.avg:.4f}) "
                  f"data_time: {data_time.latest:.2f}({data_time.avg:.2f}) "
                  f"iter_time: {iter_time.latest:.2f}({iter_time.avg:.2f}) eta: {remain_time}")

        # updata learning rate scheduler and epoch
        lr_scheduler.step()

        # log the epoch information
        logger.info(f"epoch: {epoch}/{cfg.data.epochs}, train loss: {loss_buffer.avg}, time: {epoch_timer.since_start()}s")
        iter_time.clear()
        data_time.clear()
        loss_buffer.clear()

        # write the important information into meta
        meta = {"epoch": epoch,
                "iter": iter}
    
        # save checkpoint
        checkpoint = osp.join(cfg.log_dir, "epoch_{0:05d}.pth".format(epoch))
        if (epoch == fusion_epochs) or (epoch == fusion_epochs):
            gorilla.save_checkpoint(model=model,
                                    filename=checkpoint,
                                    optimizer=optimizer,
                                    scheduler=lr_scheduler,
                                    meta=meta)
        else:
            gorilla.save_checkpoint(model=model,
                                    filename=checkpoint,
                                    meta=meta)
        logger.info("Saving " + checkpoint)
        # save as latest checkpoint
        latest_checkpoint = osp.join(cfg.log_dir, "epoch_latest.pth")
        gorilla.save_checkpoint(model=model,
                                filename=latest_checkpoint,
                                optimizer=optimizer,
                                scheduler=lr_scheduler,
                                meta=meta)

        epoch += 1


def get_checkpoint(log_dir, epoch=0, checkpoint=""):
    if not checkpoint:
        if epoch > 0:
            checkpoint = osp.join(log_dir, "epoch_{0:05d}.pth".format(epoch))
            assert osp.isfile(checkpoint)
        else:
            latest_checkpoint = glob.glob(osp.join(log_dir, "*latest*.pth"))
            if len(latest_checkpoint) > 0:
                checkpoint = latest_checkpoint[0]
            else:
                checkpoint = sorted(glob.glob(osp.join(log_dir, "*.pth")))
                if len(checkpoint) > 0:
                    checkpoint = checkpoint[-1]
                    epoch = int(checkpoint.split("_")[-1].split(".")[0])

    return checkpoint, epoch + 1

def main(args):
    # read config file
    cfg = gorilla.Config.fromfile(args.config)

    # get logger file
    log_dir, logger = gorilla.collect_logger(
        prefix=osp.splitext(osp.basename(args.config))[0])
    #### NOTE: can initlize the logger manually
    # logger = gorilla.get_logger(log_file)

    # backup the necessary file and directory(Optional, details for source code)
    backup_list = ["train.py", "test.py", "sstnet", args.config]
    backup_dir = osp.join(log_dir, "backup")
    gorilla.backup(backup_dir, backup_list)

    # merge the paramters in args into cfg
    cfg = gorilla.config.merge_cfg_and_args(cfg, args)

    cfg.log_dir = log_dir
    
    # set random seed
    seed = cfg.get("seed", 0)
    gorilla.set_random_seed(seed)

    # model
    logger.info("=> creating model ...")

    # create model
    model = gorilla.build_model(cfg.model)
    model = model.cuda()
    if args.num_gpus > 1:
        # convert the BatchNorm in model as SyncBatchNorm (NOTE: this will be error for low-version pytorch!!!)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # DDP wrap model
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gorilla.get_local_rank()], find_unused_parameters=True)

    # logger.info("Model:\n{}".format(model)) (Optional print model)

    # count the paramters of model (Optional)
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.info(f"#classifier parameters new: {count_parameters}")

    # start training
    do_train(model, cfg, logger)


if __name__ == "__main__":
    # get the args
    args = get_parser()

    # # auto using the free gpus
    # gorilla.set_cuda_visible_devices(num_gpu=args.num_gpus)

    gorilla.launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,) # use tuple to wrap
    )
