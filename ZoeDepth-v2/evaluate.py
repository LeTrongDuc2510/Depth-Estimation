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
import os
import argparse
from pprint import pprint

import torch
from torchvision.utils import save_image
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm
import numpy as np
import torch.nn as nn

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters, convert_seg_to_color, id_mapping_color_dict)



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


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if i == 216:
            continue
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth, seg_gt = sample['image'], sample['depth'], sample['segmentation']
        
        # save_image(image, "rgb.jpg")
        # save_image((depth - depth.min() / (depth.max() - depth.min()).unsqueeze(0)), "depth.jpg")

        image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        if config.get('segmentation', None):
            pred, pred_seg = infer(model, image, dataset=sample['dataset'][0], focal=focal, config=config)
        else:
            pred, _ = infer(model, image, dataset=sample['dataset'][0], focal=focal, config=config)
        # print(pred.shape, pred_seg.shape)
        
        # Save image, depth, pred for visualization
        if "save_images" in config and config.save_images:
            import os
            # print("Saving images ...")
            from PIL import Image
            import torchvision.transforms as transforms
            from zoedepth.utils.misc import colorize

            os.makedirs(config.save_images, exist_ok=True)
            # def save_image(img, path):
            

            if pred.shape[-2:] != depth.shape[-2:]:
                pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
            d = colorize(depth.squeeze().cpu().numpy(), 0, 10)
            p = colorize(pred.squeeze().cpu().numpy(), 0, 10)
            im = transforms.ToPILImage()(image.squeeze().cpu())
            im.save(os.path.join(config.save_images, f"{i}_img.png"))
            Image.fromarray(d).save(os.path.join(config.save_images, f"{i}_depth.png"))
            Image.fromarray(p).save(os.path.join(config.save_images, f"{i}_pred.png"))

            if seg_gt.shape[1] > 10:  # assuming channel is at dim 3
                    seg_gt = seg_gt.permute(0, 3, 1, 2)  # [1, 1, 480, 640]
            
            print("Pred seg shape", pred_seg.shape)
            print("GT seg shape", seg_gt.shape)
            if config.get('segmentation', None):

                
                if seg_gt.shape[-2:] != pred_seg.shape[-2:]:
                    pred_seg = nn.functional.interpolate(pred_seg, seg_gt.shape[-2:], mode='nearest')

                print("Pred seg shape after inter", pred_seg.shape)
                seg_logits = pred_seg.squeeze(0).cpu()  # Shape: [14, H, W]
                seg_mask = torch.argmax(seg_logits, dim=0).numpy().astype(np.uint8)  # Shape: [H, W]
                seg_color = convert_seg_to_color(seg_mask, id_mapping_color_dict)    # Shape: [H, W, 3]
                Image.fromarray(seg_color).save(os.path.join(config.save_images, f"{i}_seg_pred.png"))

                seg_gt = seg_gt.squeeze()
                seg_gt_color = convert_seg_to_color(seg_gt, id_mapping_color_dict)
                Image.fromarray(seg_gt_color).save(os.path.join(config.save_images, f"{i}_seg_gt.png"))

            
        # print(depth.shape, pred.shape)
        metrics.update(compute_metrics(depth, pred, config=config))

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics

def main(config):
    model = build_model(config)
    if config.get("checkpoint"):
        model.load_state_dict(torch.load(config["checkpoint"]).get("model"))
        print(f"Load ckpt from {config['checkpoint']}")

    print("Done Build model")
    test_loader = DepthDataLoader(config, 'online_eval').data
    model = model.cuda()
    metrics = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return metrics


# def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):
def eval_model(model_name, checkpoint, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    # overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **kwargs)
    config["checkpoint"] = checkpoint if os.path.exists(checkpoint) else None
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
    #                     required=False, default=None, help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-c", "--checkpoint", type=str, default='pretrained/ZoeD_M12_N.pt')
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
