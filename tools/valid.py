# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm

import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from dataset import make_test_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.vis import save_debug_images
from utils.vis import save_valid_image,return_valid_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size

torch.multiprocessing.set_sharing_strategy('file_system')


import cv2
import numpy as np

import matplotlib.pyplot as plt
import copy


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    dump_input = torch.rand(
        (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    )
    logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        # model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    data_loader, test_dataset = make_test_dataloader(cfg)

    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    parser = HeatmapParser(cfg)
    all_preds = []
    all_scores = []

    pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None

    if cfg.OWN:
        import pyrealsense2 as rs

        now_id=0
        align_to = rs.stream.color
        align = rs.align(align_to)

        #深度图帧图像滤波器
        hole_filling_filter=rs.hole_filling_filter(2)

        #配置文件
        pipe = rs.pipeline()
        cfg_rs = rs.config()
        profile = pipe.start(cfg_rs)
        # D400相机开启参数
        cfg_rs.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
        cfg_rs.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)

        try:
            while True:
                #获取帧图像
                frame = pipe.wait_for_frames()

                #对齐之后的frame
                aligned_frame = align.process(frame)

                #获得数据帧
                depth_frame = aligned_frame.get_depth_frame()
                color_frame = aligned_frame.get_color_frame()

                # 深度参数，像素坐标系转相机坐标系用到，要拿彩色作为内参，因为深度图会对齐到彩色相机
                color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                # print('color_intrin:', color_intrin)

                #将深度图彩色化的工具
                colorizer = rs.colorizer()

                #将彩色图和深度图进行numpy化
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                #输入视频
                # out.write(color_image)
                #将深度图彩色化     
                colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                all_images = np.hstack((color_image, colorized_depth))
                
                image = color_image
                base_size, center, scale = get_multi_scale_size(
                    image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
                )
                with torch.no_grad():
                    final_heatmaps = None
                    tags_list = []
                    for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                        input_size = cfg.DATASET.INPUT_SIZE
                        image_resized, center, scale = resize_align_multi_scale(
                            image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                        )
                        image_resized = transforms(image_resized)
                        image_resized = image_resized.unsqueeze(0).cuda()

                        outputs, heatmaps, tags = get_multi_stage_outputs(
                            cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                            cfg.TEST.PROJECT2IMAGE, base_size
                        )
                        

                        final_heatmaps, tags_list = aggregate_results(
                            cfg, s, final_heatmaps, tags_list, heatmaps, tags
                        )

                    final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))


                    heatmap_show_0 = np.array(final_heatmaps[0][0].cpu().detach())


                    tags = torch.cat(tags_list, dim=4)
                    grouped, scores = parser.parse(
                        final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
                    )
                    print(scores)
                    final_results = get_final_preds(
                        grouped, center, scale,
                        [final_heatmaps.size(3), final_heatmaps.size(2)]
                    )

                

                    # results = final_results  # predict on an image
                    prefix  = "output/test_output/test"
                    results_image = return_valid_image(image, final_results, '{}.jpg'.format(prefix), dataset=test_dataset.name)


                    







                # 图像展示
                # cv2.imshow('all_images', results_image)
                
                # 
                normalized_data = (heatmap_show_0 - np.min(heatmap_show_0)) / (np.max(heatmap_show_0) - np.min(heatmap_show_0)) * 255
                normalized_data = normalized_data.astype(np.uint8)
                # normalized_data_lap = cv2.Laplacian(normalized_data, cv2.CV_32F)

                _, binary = cv2.threshold(normalized_data, 50, 255, cv2.THRESH_BINARY)

                binary = binary.astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

                # 排除背景（第一个连通区域通常是背景）
                stats = stats[1:]
                centroids = centroids[1:]

                # 按面积排序，取面积最大的四个连通区域
                sorted_indices = np.argsort(stats[:, cv2.CC_STAT_AREA])[::-1]
                top_four_indices = sorted_indices[:4]

                # 提取四个亮点团的中心坐标
                centers = centroids[top_four_indices]

                # 在原图上绘制中心
                for i, center in enumerate(centers):
                    x, y = int(center[0]), int(center[1])
                    cv2.circle(normalized_data, (x, y), 5, (0, 255, 0), -1)

                resized_rgb = copy.deepcopy(cv2.resize(image, (normalized_data.shape[1],normalized_data.shape[0])))

                for i, center in enumerate(centers):
                    x, y = int(center[0]), int(center[1])
                    cv2.circle(resized_rgb, (x, y), 5, (0, 255, 0), -1)

                # 在每个连通区域中找到最亮的点
                # for index in top_four_indices:
                #     region_mask = (labels == index).astype(np.uint8)
                #     region_gray = cv2.bitwise_and(normalized_data, normalized_data, mask=region_mask)
                #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(region_gray)
                #     # 在原图上绘制最亮点
                #     cv2.circle(image, max_loc, 5, (0, 0, 255), -1)
                #     cv2.putText(image, 'Max', (max_loc[0] + 10, max_loc[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                concatenated_image = np.vstack((cv2.cvtColor(normalized_data, cv2.COLOR_GRAY2RGB), resized_rgb))
                cv2.imshow('all_images', concatenated_image)
                

                
                #帧数设定
                key = cv2.waitKey(30)

                now_id+=1



                #按键事件
                if key == ord("q"):
                    print('用户退出！')
                    break



        finally:
            pipe.stop()
        # image_path = "data/LEGO/images/train/01.png"
        image_path = "data/chatou/images/train/117.png"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )
        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = parser.parse(
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

            final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )
            prefix  = "output/test_output/test"
            save_valid_image(image, final_results, '{}.jpg'.format(prefix), dataset=test_dataset.name)
            exit()



    for i, (images, annos) in enumerate(data_loader):

        print(f"imagg {i} operated")
        assert 1 == images.size(0), 'Test batch size should be 1'

        image = images[0].cpu().numpy()
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )

        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = parser.parse(
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

            final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )

        if cfg.TEST.LOG_PROGRESS:
            pbar.update()

        if i % cfg.PRINT_FREQ == 0:
            prefix = '{}_{}'.format(os.path.join(final_output_dir, 'result_valid'), i)
            # logger.info('=> write {}'.format(prefix))
            save_valid_image(image, final_results, '{}.jpg'.format(prefix), dataset=test_dataset.name)
            # save_debug_images(cfg, image_resized, None, None, outputs, prefix)

        all_preds.append(final_results)
        all_scores.append(scores)

    if cfg.TEST.LOG_PROGRESS:
        pbar.close()

    name_values, _ = test_dataset.evaluate(
        cfg, all_preds, all_scores, final_output_dir
    )

    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(logger, name_value, cfg.MODEL.NAME)
    else:
        _print_name_value(logger, name_values, cfg.MODEL.NAME)


if __name__ == '__main__':
    main()
