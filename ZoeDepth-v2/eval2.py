# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import argparse
from pprint import pprint

import torch
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters, color_label)


@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    def get_segmentation_from_prediction(pred):
        return pred['segmentation']

    config = kwargs['config']
    pred1 = model(images, **kwargs)
    if config.get('segmentation', None):
        seg1 = get_segmentation_from_prediction(pred1)
        
    pred1 = get_depth_from_prediction(pred1)
    if pred1.ndim == 3:
        pred1 = pred1.unsqueeze(1)
        
    pred2 = model(torch.flip(images, [3]), **kwargs)
    if config.get('segmentation', None):
        seg2 = get_segmentation_from_prediction(pred2)
        
    pred2 = get_depth_from_prediction(pred2)
    if pred2.ndim == 3:
        pred2 = pred2.unsqueeze(1)

    if config.get('segmentation', None):
        seg2 = torch.flip(seg2, [3])
        mean_seg = 0.5 * (seg1 + seg2)
        
    pred2 = torch.flip(pred2, [3])
    # pred2 = torch.flip(pred2, [2])

    mean_pred = 0.5 * (pred1 + pred2)

    if config.get('segmentation', None):
        return mean_pred, mean_seg
    else:
        return mean_pred, None

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
def compute_scaled_prediction(prediction, target, mask, interpolate=True, return_interpolated=False):
    if target.ndim == 3:
        target = target.unsqueeze(1)

    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(1)
    
    
    if prediction.shape[-1] != target.shape[-1] and interpolate:
        prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
        intr_input = prediction
    else:
        intr_input = prediction


    prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
    if target.ndim == 2:
        target = target.unsqueeze(0)

    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
        
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        
    assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

    scale, shift = compute_scale_and_shift(prediction, target, mask)

    scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
    return scaled_prediction

@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        # if i == 216:
        #     continue
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth = sample['image'], sample['depth']
        mask = sample["mask"].cuda().to(torch.bool)
        image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        if config.get('segmentation', None):
            seg_gt = sample['segmentation'].cuda()
        else:
            seg_gt = None
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        pred, pred_seg = infer(model, image, dataset=sample['dataset'][0], focal=focal, config=config)

        # print(pred.max(), pred.min())
        # print(depth[mask].max(), depth[mask].min())
        import torchvision.utils as vutils
        if "relative" in config and config.relative:
            if config["dataset"] == 'nyu':
                gt = depth.clone()
                depth[mask] = 1./(depth[mask] + 1e-6)
                depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min())
            else:
                gt = sample['gt'].cuda().squeeze().unsqueeze(0).unsqueeze(0)

            pred = compute_scaled_prediction(pred, depth, mask)
            vutils.save_image(pred, 'pred_disp.jpg')
            vutils.save_image(depth, 'gt_disp.jpg')
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)

            ############## Comment these lines for visualization ##############
            # depth = gt
            # disp = gt
            # disp = 1/(gt + 1e-6)

            # pred = torch.clamp(pred, min=0.0, max=1.0)
            # pred[mask] = pred[mask] * (disp[mask].max() - disp[mask].min()) + disp[mask].min()
            # pred[mask] = 1./pred[mask] - 1e-6
            ##################################################################

        if i % 10 == 0:
            # print(pred[mask].max(), pred[mask].min())
            # print(depth[mask].max(), depth[mask].min())
            # vutils.save_image((pred - pred.min())/(pred.max() - pred.min()), 'prediction.jpg')
            vutils.save_image(pred/10.0, 'prediction.jpg')
            # vutils.save_image(gt/10.0, 'gt.jpg')
            vutils.save_image(depth/10.0, 'depth.jpg')
            if seg_gt is not None:
                vutils.save_image(torch.argmax(pred_seg, dim=1)/13.0, 'pred_seg.jpg')
                vutils.save_image(seg_gt.squeeze(-1)/13.0, 'seg_gt.jpg')
    
        # Save image, depth, pred for visualization
        if "save_images" in config and config.save_images:
            import os
            # print("Saving images ...")
            from PIL import Image
            import torchvision.transforms as transforms
            from zoedepth.utils.misc import colorize

            os.makedirs(config.save_images, exist_ok=True)
            # def save_image(img, path):
            if "relative" in config and config.relative:
                d = colorize(depth.squeeze().cpu().numpy(), 0, 1, cmap='gray_r')
                p = colorize((0.9*pred+0.1*depth).squeeze().cpu().numpy(), 0, 1, cmap='gray_r')
            else:
                d = colorize(depth.squeeze().cpu().numpy(), 0, 10, cmap='gray_r')
                p = colorize(pred.squeeze().cpu().numpy(), 0, 10, cmap='gray_r')
            im = transforms.ToPILImage()(image.squeeze().cpu())
            im.save(os.path.join(config.save_images, f"{i}_img.png"))
            Image.fromarray(d).save(os.path.join(config.save_images, f"{i}_depth.png"))
            Image.fromarray(p).save(os.path.join(config.save_images, f"{i}_pred.png"))
            if seg_gt is not None:
                s = color_label(torch.argmax(pred_seg, dim=1).squeeze().cpu().numpy(), num_class=13)
                Image.fromarray(s).save(os.path.join(config.save_images, f"{i}_seg_pred.png"))
                
                s_gt = color_label(seg_gt.squeeze().cpu().numpy(), num_class=13)
                Image.fromarray(s_gt).save(os.path.join(config.save_images, f"{i}_seg_gt.png"))
        # print(depth.shape, pred.shape)
        metrics.update(compute_metrics(depth, pred, seg_gt, pred_seg, num_classes=14, config=config))

        # if i == 10:
        #     break

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics

def main(config):
    model = build_model(config)
    test_loader = DepthDataLoader(config, 'online_eval').data
    model = model.cuda()
    metrics = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return metrics


def eval_model(model_name, checkpoint, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "checkpoint": checkpoint} if checkpoint else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    # parser.add_argument("-p", "--pretrained_resource", type=str,
    #                     required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, default="")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if "ALL_INDOOR" in args.dataset:
        datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset:
        datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset:
        datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, checkpoint=args.checkpoint,
                    dataset=dataset, **overwrite_kwargs)
