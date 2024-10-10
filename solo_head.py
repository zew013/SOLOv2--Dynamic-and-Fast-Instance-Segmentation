import matplotlib.pyplot as plt
import torchvision
from torch import optim
import torch
from torch import nn
import torch.nn.functional as F
from solo_arc import Category, Mask
from torchvision import transforms
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import os
from functools import partial

class SOLOHead(pl.LightningModule):
    _default_cfg = {
        'num_classes': 4,
        'in_channels': 256,
        'seg_feat_channels': 256,
        'stacked_convs': 7,
        'strides': [8, 8, 16, 32, 32],
        'scale_ranges': [(1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)],
        'epsilon': 0.2,
        'num_grids': [40, 36, 24, 16, 12],
        'mask_loss_cfg': dict(weight=3),
        'cate_loss_cfg': dict(gamma=2, alpha=0.25, weight=1),
        'postprocess_cfg': dict(cate_thresh=0.2, mask_thresh=0.5, pre_NMS_num=50, keep_instance=5, IoU_thresh=0.5)
    }
   
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in {**self._default_cfg, **kwargs}.items():
            setattr(self, k, v)

        pretrained_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)
        self.backbone = pretrained_model.backbone
        self.num_levels = len(self.scale_ranges)
        self.scale_ranges = torch.tensor(self.scale_ranges).to(self.device)
        self.cat_branch = Category()
        self.mask_branch = Mask(self.num_grids)
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]


    
    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    def new_FPN(self, fpn_feat_list):
        fpn_feat_list[0] = F.interpolate(fpn_feat_list[0], scale_factor=0.5,
                mode='bilinear') 
        fpn_feat_list[-1] = F.interpolate(fpn_feat_list[-1], (25, 34),  
                mode='bilinear') 
        return fpn_feat_list
    
    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        ## TODO: finish forward function for single level in FPN.
        ## Notice, we distinguish the training and inference.
        category_prediction = fpn_feat
        instance_prediction = fpn_feat
        current_level_grid = self.num_grids[idx]  # current level grid

        resized_feature_map = F.interpolate(fpn_feat, size=(current_level_grid, current_level_grid), mode='nearest')
        category_prediction = self.cat_branch(resized_feature_map, idx)
        pixel_coords_x, pixel_coords_y = self.generate_pixel_coordinates(fpn_feat.shape[-2:])
        instance_prediction = self.mask_branch(fpn_feat, pixel_coords_x, pixel_coords_y, idx)
        
        # In inference time, upsample the prediction to (original image size / 4)
        if eval:
            category_prediction = self.points_nms(category_prediction).permute(0, 2, 3, 1)
            instance_prediction = F.interpolate(instance_prediction, (200, 272), mode='nearest')

        # Check flag for training vs inference
        if not eval:
            instance_prediction = F.interpolate(instance_prediction, scale_factor=2, mode='nearest')
            assert category_prediction.shape[1:] == (3, current_level_grid, current_level_grid)
            assert instance_prediction.shape[1:] == (current_level_grid ** 2, fpn_feat.shape[2] * 2, fpn_feat.shape[3] * 2)

        return category_prediction, instance_prediction
    
    def forward(self, images, eval=False):
        # You can modify this if you want to train the backbone
        backbone_features = [feature.detach() for feature in self.backbone(images).values()]
        
        feature_pyramid_list = self.new_FPN(backbone_features)

        category_predictions_list = []
        instance_predictions_list = []
        for i in range(5): 
            cate_pred, ins_pred = self.forward_single_level(backbone_features[i], i, eval)
            category_predictions_list.append(cate_pred)
            instance_predictions_list.append(ins_pred)

        # Check flag for training vs inference
        if not eval:
            assert category_predictions_list[1].shape[2] == self.num_grids[1]
            assert instance_predictions_list[1].shape[1] == self.num_grids[1] ** 2

        return category_predictions_list, instance_predictions_list
    
    def generate_pixel_coordinates(self, size):
        height, width = size
        i_coord_values = torch.linspace(-1, 1, height).unsqueeze(-1)
        j_coord_values = torch.linspace(-1, 1, width).unsqueeze(0)

        i_coord_matrix = i_coord_values.repeat(1, width)
        j_coord_matrix = j_coord_values.repeat(height, 1)

        i_coord_tensor = i_coord_matrix.unsqueeze(0).unsqueeze(0)
        j_coord_tensor = j_coord_matrix.unsqueeze(0).unsqueeze(0)

        i_coord_tensor = i_coord_tensor.to(self.device)
        j_coord_tensor = j_coord_tensor.to(self.device)

        return i_coord_tensor, j_coord_tensor

    def compute_centre_regions(self, boxes_img, heights, widths):
        center_y = (boxes_img[:, 3] + boxes_img[:, 1])/2
        center_x = (boxes_img[:, 2] + boxes_img[:, 0])/2
            
        x1 = center_x - self.epsilon * widths / 2
        y1 = center_y - self.epsilon * heights / 2    
        x2 = center_x + self.epsilon * widths / 2
        y2 = center_y + self.epsilon * heights / 2
       
        centre_regions = torch.column_stack([x1, y1, x2, y2])
        
        return centre_regions

    def generate_targets(self, bounding_boxes, labels, masks):
        category_targets = []
        mask_targets = []
        active_masks = []

        category_targets, mask_targets, active_masks = self.MultiApply(self.generate_target_per_img, bounding_boxes, labels, masks)
        return category_targets, mask_targets, active_masks


    def generate_target_per_img(self, boxes_img, labels_img, masks_img):
        # Extract the height and width of the input image
        image_height, image_width = masks_img.shape[-2:]

        # Calculate bounding box heights and widths
        box_heights = boxes_img[:, 3] - boxes_img[:, 1]
        box_widths = boxes_img[:, 2] - boxes_img[:, 0]

        # Compute center regions and areas for FPN level assignment
        center_regions = self.compute_centre_regions(boxes_img, box_heights, box_widths)
        box_sqrt_areas = (box_heights * box_widths).sqrt()

        scale_range = self.scale_ranges.to(self.device)

        # Determine FPN level assignment masks for each bounding box based on its area
        fpn_level_masks = (box_sqrt_areas.unsqueeze(dim=1) >= scale_range[:, 0]) & (box_sqrt_areas.unsqueeze(dim=1) <= scale_range[:, 1])

        # Initialize lists to hold target values for each feature pyramid level
        category_targets_per_img = []
        active_masks_per_img = []
        mask_targets_per_img = []

        # Loop through each feature pyramid level
        for level_index in range(self.num_levels):
            # Initialize empty tensors for current feature pyramid level
            level_grid_size = self.num_grids[level_index]
            active_mask = torch.zeros((level_grid_size, level_grid_size), dtype=torch.bool).to(self.device)
            category_target = torch.zeros((level_grid_size, level_grid_size), dtype=torch.uint8).to(self.device)
            
            # Compute feature map dimensions for the current level
            feature_height = image_height // self.strides[level_index]
            feature_width = image_width // self.strides[level_index]
            mask_target = torch.zeros(level_grid_size, level_grid_size, 2 * feature_height, 2 * feature_width).to(self.device)

            # Calculate grid cell dimensions for the current level
            grid_cell_width = image_width / level_grid_size
            grid_cell_height = image_height / level_grid_size

            # Filter boxes, labels, and masks for the current feature pyramid level
            filtered_center_regions = center_regions[fpn_level_masks[:, level_index]]
            filtered_labels = labels_img[fpn_level_masks[:, level_index]]
            filtered_masks = masks_img[fpn_level_masks[:, level_index]]

            # If no valid regions exist for the current level, append empty results and continue to next level
            if len(filtered_center_regions) == 0:
                active_masks_per_img.append(active_mask.reshape(-1))
                category_targets_per_img.append(category_target)
                mask_targets_per_img.append(mask_target.reshape(-1, 2 * feature_height, 2 * feature_width))
                continue

            # Calculate grid coordinates for filtered regions
            x_match_start = (filtered_center_regions[:, 0] / grid_cell_width).floor().int()
            x_match_end = (filtered_center_regions[:, 2] / grid_cell_width).floor().int()
            y_match_start = (filtered_center_regions[:, 1] / grid_cell_height).floor().int()
            y_match_end = (filtered_center_regions[:, 3] / grid_cell_height).floor().int()

            # Populate category and mask targets for the current level
            for box_index in range(len(x_match_start)):
                x1 = x_match_start[box_index]
                x2 = x_match_end[box_index]
                y1 = y_match_start[box_index]
                y2 = y_match_end[box_index]

                # Update active mask and category target for the grid cells corresponding to the current box
                active_mask[y1:y2 + 1, x1:x2 + 1] = 1
                category_target[y1:y2 + 1, x1:x2+ 1] = filtered_labels[box_index]

                # Resize the corresponding instance mask and update the mask target
                resized_mask = F.interpolate(filtered_masks[box_index].view(1, 1, image_height, -1), size=(2 * feature_height, 2 * feature_width), mode='nearest')
                mask_target[y1:y2 + 1, x1:x2 + 1] = resized_mask

            # Append results for the current level
            active_masks_per_img.append(active_mask.reshape(-1))
            category_targets_per_img.append(category_target)
            mask_targets_per_img.append(mask_target.reshape(-1, 2 * feature_height, 2 * feature_width))

        return category_targets_per_img, mask_targets_per_img, active_masks_per_img    
    
    def PlotGT(self, category_targets, mask_targets, active_masks, color_list, img):
        # Loop through images in the batch
        for image_index in range(len(mask_targets)):
            # Initialize a figure for the current image with one row and five columns (for five FPN layers)
            fig, axes = plt.subplots(1, 5, figsize=(20, 5))
            image = img[image_index].permute(1, 2, 0)  # Convert image from CxHxW to HxWxC

            # Normalize the image for visualization (scale between [0, 1])
            min_val, max_val = image.min(), image.max()
            normalized_image = (image - min_val) / (max_val - min_val)

            # Loop through each feature pyramid layer
            for layer_index in range(len(mask_targets[image_index])):
                ax = axes[layer_index]  # Use the corresponding subplot for each FPN layer
                ax.imshow(normalized_image)  # Plot the normalized image
                ax.axis('off')  # Hide axis for a cleaner view
                ax.set_title(f"Layer {layer_index + 1}")

                # Loop through the three categories (vehicles, people, animals)
                for category_index in range(3):
                    color = color_list[category_index]
                    category_indices = torch.where(category_targets[image_index][layer_index] == (category_index + 1))

                    # Skip if no indices are found for the current category
                    if len(category_indices[0]) == 0:
                        continue

                    # Flatten the indices to retrieve corresponding mask targets
                    flat_indices = category_indices[0] * self.num_grids[layer_index] + category_indices[1]
                    selected_masks = mask_targets[image_index][layer_index][flat_indices]

                    # Loop through each selected mask and overlay it on the image
                    for mask in selected_masks:
                        # Add batch and channel dimensions, then interpolate to original image size
                        upsampled_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(image.shape[0], image.shape[1]), mode='bilinear')
                        upsampled_mask = torch.squeeze(upsampled_mask)

                        # Binarize the mask and mask zero values for visualization
                        upsampled_mask[upsampled_mask > 0] = 1
                        binary_mask = np.ma.masked_where(upsampled_mask == 0, upsampled_mask)

                        # Overlay the mask with full opacity
                        ax.imshow(binary_mask, cmap=color, alpha=1)

            # Adjust layout to prevent overlap and show the plot for the current image
            plt.tight_layout()
            plt.show()
  

    def category_loss(self, pred, target):
        batch_size = pred.shape[0]
        grid_size = pred.shape[-1]
        alpha = self.cate_loss_cfg['alpha']
        gamma = self.cate_loss_cfg['gamma']

        target_onehot = F.one_hot(torch.stack(target).to(torch.int64), num_classes=4)
        target_onehot = target_onehot.permute(0, 3, 1, 2)
        target_onehot = target_onehot[:, 1:, :, :]

        prob = pred * target_onehot + (1 - pred) * (1 - target_onehot)
        alpha = alpha * target_onehot + (1 - alpha) * (1 - target_onehot)

        focal_loss = -alpha * torch.log(prob + 1e-9) * (1 - prob) ** gamma
        loss = focal_loss.sum() / (3 * grid_size * grid_size) / batch_size

        return loss 
    
    def mask_loss(self, pred, target, active_mask):
        batch_size = pred.shape[0]
        active_mask_indices = torch.where(active_mask)
        counts = len(active_mask_indices[0])

        if counts == 0:
            return 0

        intersection = 2 * (pred[active_mask_indices] * target[active_mask_indices]).sum(axis=[1, 2])
        total_area = (pred[active_mask_indices]**2).sum(axis=[1, 2]) + (target[active_mask_indices] ** 2).sum(axis=[1, 2])

        loss = 1 - intersection / total_area

        return loss.sum() / counts

    def loss(self,
             cate_pred_list,
             mask_pred_list,
             mask_targets_list,
             active_masks_list,
             cate_targets_list):
        
        num_level = len(self.num_grids)
        total_cate_loss = 0
        total_mask_loss = 0
        for i in range(num_level):
            cate_targets_per_level = [cate_targets[i] for cate_targets in cate_targets_list]
            cate_loss = self.category_loss(cate_pred_list[i], cate_targets_per_level)   
            total_cate_loss += cate_loss
            
            mask_targets_per_level = torch.stack([mask_targets[i] for mask_targets in mask_targets_list])
            active_masks_per_level = torch.stack([active_masks[i] for active_masks in active_masks_list])
            mask_loss = self.mask_loss(mask_pred_list[i], mask_targets_per_level, active_masks_per_level)
   
            total_mask_loss += mask_loss
        return self.cate_loss_cfg['weight'] * total_cate_loss, self.mask_loss_cfg['weight'] * total_mask_loss
  
    def training_step(self, batch, batch_idx):
          imgs, labels, masks, bboxes = batch
          batch_size = imgs.size(0)
          category_targets, mask_targets, active_masks = self.generate_targets(bboxes, labels, masks)
          cate_pred_list, ins_pred_list = self(imgs)
          
          total_cate_loss, total_mask_loss = self.loss(cate_pred_list, ins_pred_list, mask_targets, active_masks, category_targets)
          total_loss = total_cate_loss + total_mask_loss
          self.log("loss", total_loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=batch_size)
          self.log("train_category_loss", total_cate_loss, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
          self.log("train_mask_loss", total_mask_loss, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
          return total_loss
      
    def validation_step(self, batch, batch_idx):
        imgs, labels, masks, bboxes = batch
        batch_size = imgs.size(0)
        category_targets, mask_targets, active_masks = self.generate_targets(bboxes, labels, masks)
        cate_pred_list, ins_pred_list = self(imgs)
        
        total_cate_loss, total_mask_loss = self.loss(cate_pred_list, ins_pred_list, mask_targets, active_masks, category_targets)
        total_val_loss = total_cate_loss + total_mask_loss
        self.log("val_loss", total_val_loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=batch_size)
        self.log("val_category_loss", total_cate_loss, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
        self.log("val_mask_loss", total_mask_loss, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
        return total_val_loss
      
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=[27, 33], gamma=0.1),
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    
    def points_nms(self, heat, kernel=2):
        # Input:  (batch_size, C-1, S, S)
        # Output: (batch_size, C-1, S, S)
        # kernel must be 2
        hmax = F.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep





























    


