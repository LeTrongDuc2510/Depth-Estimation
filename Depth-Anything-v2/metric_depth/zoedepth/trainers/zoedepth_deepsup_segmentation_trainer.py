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

import torch
import torch.cuda.amp as amp
import torch.nn as nn

from zoedepth.trainers.loss import GradL1Loss, SILogLoss, ScaleAndShiftInvariantLoss, DeepSupervisedLoss, CombinedSegmentationLoss, DeepSupervisionSegmentationLoss
from zoedepth.utils.config import DATASETS_CONFIG
from zoedepth.utils.misc import compute_metrics
from zoedepth.data.preprocess import get_black_border

from .base_trainer import BaseTrainer
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.utils as vutils

class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device)
        self.device = device
        # self.silog_loss = ScaleAndShiftInvariantLoss()
        self.grad_loss = GradL1Loss()
        if config.relative:
            self.silog_loss = ScaleAndShiftInvariantLoss()
            self.deepsup_loss = DeepSupervisedLoss(scales=4, reduction='batch-based', loss=ScaleAndShiftInvariantLoss())
        else:    
            self.silog_loss = SILogLoss()
            self.deepsup_loss = DeepSupervisedLoss(scales=4, reduction='batch-based', loss=SILogLoss())
        self.segmentation_loss = CombinedSegmentationLoss()
        self.deepsup_segmentation_loss = DeepSupervisionSegmentationLoss(scales=4, reduction='batch-based', loss=CombinedSegmentationLoss())
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)
        self.accumulation_steps = self.config.desired_batch_size // self.config.batch_size
        self.optimizer.zero_grad() # Initialize gradients here

    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w
        batch["depth"].shape : batch_size, 1, h, w
        """
        images, depths_gt, segmentations_gt = batch['image'].to(
            self.device), batch['depth'].to(self.device), batch['segmentation'].to(self.device)
        dataset = batch['dataset'][0]

        b, c, h, w = images.size()
        mask = batch["mask"].to(self.device).to(torch.bool)

        losses = {}

        with amp.autocast(enabled=self.config.use_amp):
            output = self.model(images)
            pred_depths = output['metric_depth']

            l_si, pred = self.silog_loss(
                pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True)
            loss = self.config.w_si * l_si
            losses[self.silog_loss.name] = self.config.w_si * l_si

            l_deep_sup = self.deepsup_loss(output['deep_sup'], depths_gt, mask=mask, normalized_target_= not self.config.relative)
            loss = loss + self.config.w_deep_sup * l_deep_sup
            losses[self.deepsup_loss.name] = self.config.w_deep_sup * l_deep_sup

            l_segmentation = self.segmentation_loss(output['segmentation'], segmentations_gt)
            loss = loss + self.config.w_segmentation * l_segmentation
            losses[self.segmentation_loss.name] = self.config.w_segmentation * l_segmentation

            l_deep_sup_segmentation = self.deepsup_segmentation_loss(output['deep_sup_segmentation'], segmentations_gt)
            loss = loss + self.config.w_deep_sup_segmentation * l_deep_sup_segmentation
            losses[self.deepsup_segmentation_loss.name] = self.config.w_deep_sup_segmentation * l_deep_sup_segmentation

            if self.config.w_grad > 0:
                l_grad = self.grad_loss(pred, depths_gt, mask=mask)
                loss = loss + self.config.w_grad * l_grad
                losses[self.grad_loss.name] = self.config.w_grad * l_grad

        if train_step % 50 == 0:
            if not self.config.relative:
                vutils.save_image(pred_depths/10.0, 'prediction.png')
                vutils.save_image(depths_gt/10.0, 'target.png')
            else:
                vutils.save_image((pred_depths.unsqueeze(1) - pred_depths.min())/(pred_depths.max() - pred_depths.min()), 'prediction.png')
                vutils.save_image((depths_gt - depths_gt.min())/(depths_gt.max() - depths_gt.min()), 'target.png')
                # vutils.save_image(pred_depths, 'prediction.png')
                # vutils.save_image(depths_gt, 'target.png')
                
            vutils.save_image(torch.argmax(output['segmentation'], dim=1, keepdim=True) / 13.0, 'prediction_segmentation.png')
            vutils.save_image(segmentations_gt / 13.0, 'segmentation_gt.png')
            vutils.save_image(mask.float(), 'mask.png')
            vutils.save_image(images, 'rgb.png', normalize=True, scale_each=True)

        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=self.device)
            print(f"SSI Loss returns NaN")
        self.scaler.scale(loss / self.accumulation_steps).backward()

        if (train_step + 1) % self.accumulation_steps == 0:
            if self.config.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip_grad)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        if self.should_log and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
            # -99 is treated as invalid depth in the log_images function and is colored grey.
            depths_gt_log = depths_gt.clone()
            depths_gt_log[torch.logical_not(mask)] = -99

            self.log_images(rgb={"Input": images[0, ...]}, depth={"GT": depths_gt_log[0], "PredictedMono": pred[0]}, prefix="Train",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

            if self.config.get("log_rel", False):
                self.log_images(
                    scalar_field={"RelPred": output["relative_depth"][0]}, prefix="TrainRel")

        return losses
    
    @torch.no_grad()
    def eval_infer(self, x):
        with amp.autocast(enabled=self.config.use_amp):
            m = self.model.module if self.config.multigpu else self.model
            outs = m(x)
            pred_depths = outs['metric_depth']
            pred_seg = outs['segmentation']

        return pred_depths, pred_seg

    @torch.no_grad()
    def crop_aware_infer(self, x):
        # if we are not avoiding the black border, we can just use the normal inference
        if not self.config.get("avoid_boundary", False):
            return self.eval_infer(x)
        
        # otherwise, we need to crop the image to avoid the black border
        # For now, this may be a bit slow due to converting to numpy and back
        # We assume no normalization is done on the input image

        # get the black border
        assert x.shape[0] == 1, "Only batch size 1 is supported for now"
        x_pil = transforms.ToPILImage()(x[0].cpu())
        x_np = np.array(x_pil, dtype=np.uint8)
        black_border_params = get_black_border(x_np)
        top, bottom, left, right = black_border_params.top, black_border_params.bottom, black_border_params.left, black_border_params.right
        x_np_cropped = x_np[top:bottom, left:right, :]
        x_cropped = transforms.ToTensor()(Image.fromarray(x_np_cropped))

        # run inference on the cropped image
        pred_depths_cropped, pred_seg = self.eval_infer(x_cropped.unsqueeze(0).to(self.device))

        # resize the prediction to x_np_cropped's size
        pred_depths_cropped = nn.functional.interpolate(
            pred_depths_cropped, size=(x_np_cropped.shape[0], x_np_cropped.shape[1]), mode="bilinear", align_corners=False)
        

        # pad the prediction back to the original size
        pred_depths = torch.zeros((1, 1, x_np.shape[0], x_np.shape[1]), device=pred_depths_cropped.device, dtype=pred_depths_cropped.dtype)
        pred_depths[:, :, top:bottom, left:right] = pred_depths_cropped

        return pred_depths, pred_seg

    def validate_on_batch(self, batch, val_step):
        images = batch['image'].to(self.device)
        depths_gt = batch['depth'].to(self.device)
        seg_gt = batch['segmentation'].to(self.device)
        dataset = batch['dataset'][0]
        mask = batch["mask"].to(self.device)
        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth']:
                return None, None

        if dataset == 'nyu':
            pred_depths, pred_seg = self.crop_aware_infer(images)
        else:
            pred_depths, pred_seg = self.eval_infer(images)

        depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
        seg_gt = seg_gt.squeeze().unsqueeze(0).unsqueeze(0)
        mask = mask.squeeze().unsqueeze(0).unsqueeze(0)
        pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)
        
        depths_gt = depths_gt.detach()
        seg_gt = seg_gt.detach()
        mask = mask.detach()
        pred_depths = pred_depths.detach()
        
        with amp.autocast(enabled=self.config.use_amp):
            l_depth = self.config.w_si * self.silog_loss(
                pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)
            
            l_segmentation = self.config.w_segmentation * self.segmentation_loss(pred_seg, seg_gt)

        
        metrics = compute_metrics(depths_gt, pred_depths, seg_gt, pred_seg, num_classes=14, **self.config)
        losses = {f"{self.silog_loss.name}": l_depth.item(), f"{self.segmentation_loss.name}": l_segmentation.item()}

        if val_step == 1 and self.should_log:
            depths_gt[torch.logical_not(mask)] = -99
            self.log_images(rgb={"Input": images[0]}, depth={"GT": depths_gt[0], "PredictedMono": pred_depths[0]}, prefix="Test",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

        return metrics, losses