import torch
import torch.cuda.amp as amp
import torch.nn as nn

from zoedepth.trainers.loss import GradL1Loss, SILogLoss, ScaleAndShiftInvariantLoss, MultiscaleGradientLoss
from zoedepth.utils.config import DATASETS_CONFIG
from zoedepth.utils.misc import compute_metrics
from zoedepth.data.preprocess import get_black_border

from .base_trainer import BaseTrainer
from torchvision import transforms, utils as vutils
from PIL import Image
import numpy as np

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            print(f"Gradient norm for {name}: {grad_norm}")
            if grad_norm > 100:  # This can indicate exploding gradients
                print(f"Warning: Exploding gradients in {name}")
            elif grad_norm < 1e-6:  # This can indicate vanishing gradients
                print(f"Warning: Vanishing gradients in {name}")

class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device)
        self.device = device
        # self.silog_loss = SILogLoss()
        # self.grad_loss = GradL1Loss()
        self.ssi_loss = ScaleAndShiftInvariantLoss()
        self.msi_loss = MultiscaleGradientLoss(scales=4, reduction='batch-based')
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)
        self.accum_steps = self.config.accumulation_steps

    def train_on_batch(self, batch, train_step, is_last_batch = False):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w
        batch["depth"].shape : batch_size, 1, h, w
        """

        images, depths_gt = batch['image'].to(
            self.device), batch['depth'].to(self.device)
        dataset = batch['dataset'][0]
        

        b, c, h, w = images.size()
        mask = batch["mask"].to(self.device).to(torch.bool)

        print(f"Batch shapes: image {images.shape}, depth {depths_gt.shape}, mask {mask.shape}")
        # print("Valid mask ratio:", mask.sum().item() / mask.numel())

        losses = {}

        with amp.autocast(enabled=self.config.use_amp):

            output = self.model(images)
            # pred_depths = output['metric_depth']
            pred_depths = output['rel_depth']

            l_ssi, pred = self.ssi_loss(
                pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True)
            loss = self.config.w_ssi * l_ssi
            losses[self.ssi_loss.name] = l_ssi
            if self.config.w_msi > 0:
                l_msi = self.msi_loss(pred, depths_gt, mask=mask)
                loss = loss + self.config.w_msi * l_msi
                losses[self.msi_loss.name] = l_msi
            else:
                l_msi = torch.Tensor([0])

            print("losses", losses)

        if train_step == 0: # ensure fresh start
            self.optimizer.zero_grad()

        if train_step % 50 == 0:
            vutils.save_image(pred, 'prediction.png', normalize=True, scale_each=True)
            vutils.save_image(depths_gt, 'depth_gt.png', normalize=True, scale_each=True)
            vutils.save_image(mask.float(), 'mask.png')
            vutils.save_image(images, 'rgb.png', normalize=True, scale_each=True)
        
        self.scaler.scale(loss).backward()
        # check_gradients(self.model)
        if self.config.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad)

        should_step = ((train_step + 1) % self.accum_steps) == 0 or is_last_batch # step every accumulation_steps or at the last batch
        # print(f"Step: {train_step}, Accumulated: {should_step}")
        if should_step:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        # self.scaler.step(self.optimizer)        

        if self.should_log and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
            # -99 is treated as invalid depth in the log_images function and is colored grey.
            depths_gt[torch.logical_not(mask)] = -99

            self.log_images(rgb={"Input": images[0, ...]}, depth={"GT": depths_gt[0], "PredictedMono": pred[0]}, prefix="Train",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

            if self.config.get("log_rel", False):
                self.log_images(
                    scalar_field={"RelPred": output["relative_depth"][0]}, prefix="TrainRel")

        # self.scaler.update()
        # self.optimizer.zero_grad()

        return losses
    
    @torch.no_grad()
    def eval_infer(self, x):
        with amp.autocast(enabled=self.config.use_amp):
            m = self.model.module if self.config.multigpu else self.model
            # pred_depths = m(x)['metric_depth']
            pred_depths = m(x)['rel_depth']
        return pred_depths

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
        pred_depths_cropped = self.eval_infer(x_cropped.unsqueeze(0).to(self.device))

        # resize the prediction to x_np_cropped's size
        pred_depths_cropped = nn.functional.interpolate(
            pred_depths_cropped, size=(x_np_cropped.shape[0], x_np_cropped.shape[1]), mode="bilinear", align_corners=False)
        

        # pad the prediction back to the original size
        pred_depths = torch.zeros((1, 1, x_np.shape[0], x_np.shape[1]), device=pred_depths_cropped.device, dtype=pred_depths_cropped.dtype)
        pred_depths[:, :, top:bottom, left:right] = pred_depths_cropped

        return pred_depths



    def validate_on_batch(self, batch, val_step):
        images = batch['image'].to(self.device)
        depths_gt = batch['depth'].to(self.device)
        dataset = batch['dataset'][0]
        mask = batch["mask"].to(self.device)
        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth']:
                return None, None

        depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
        mask = mask.squeeze().unsqueeze(0).unsqueeze(0)
        if dataset == 'nyu':
            pred_depths = self.crop_aware_infer(images)
        else:
            pred_depths = self.eval_infer(images)
        pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)

        # calculate both losses
        with amp.autocast(enabled=self.config.use_amp):
            # l_depth = self.silog_loss(
                # pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)
            l_depth_ssi, pred = self.ssi_loss(
                pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True, return_interpolated=True)
            l_depth_msi = self.msi_loss(
                pred, depths_gt, mask=mask.to(torch.bool))
            l_depth_msi = self.config.w_msi * l_depth_msi
            l_depth = l_depth_ssi + l_depth_msi

        metrics = compute_metrics(depths_gt, pred_depths, **self.config)
        # losses = {f"{self.silog_loss.name}": l_depth.item()}
        losses = {f"{self.ssi_loss.name}": l_depth_ssi.item(),
                  f"{self.msi_loss.name}": l_depth_msi.item(),
                  "TotalLoss": l_depth.item()}

        if val_step == 1 and self.should_log:
            depths_gt[torch.logical_not(mask)] = -99
            self.log_images(rgb={"Input": images[0]}, depth={"GT": depths_gt[0], "PredictedMono": pred_depths[0]}, prefix="Test",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

        return metrics, losses
