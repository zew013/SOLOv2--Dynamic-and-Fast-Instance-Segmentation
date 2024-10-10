## Author: Lishuo Pan 2020/4/18

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # TODO: load dataset, make mask list
        img_p, mask_p, label_p, bbox_p = path
        self.imgs = h5py.File(img_p, 'r')['data']

        self.masks = h5py.File(mask_p, 'r')['data']

        self.transform = transforms.Compose([
            transforms.Resize((800, 1066)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Pad((11, 0)) 
        ])
        self.labels = np.load(label_p, allow_pickle=True)
        self.bboxes = np.load(bbox_p, allow_pickle=True)

        self.cumulative_label_counts = np.cumsum([0] + [len(label) for label in self.labels])


    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, index):
        # TODO: __getitem__
        raw_img = self.imgs[index]
        mask_init_index = self.cumulative_label_counts[index]
        mask_end_index = self.cumulative_label_counts[index+1] 
        raw_mask = self.masks[mask_init_index:mask_end_index]  
        raw_bbox = self.bboxes[index]
        label = self.labels[index]
        # Preprocess the raw data
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(raw_img, raw_mask, raw_bbox)
        label = torch.tensor(label)
        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox
    def __len__(self):
        return len(self.labels)

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # TODO: image preprocess
        original_width = img.shape[-1]
        img = img /255.0
        img = torch.tensor(img, dtype=torch.float32)
        img = self.transform(img)

        mask = mask.astype(np.float32)
        mask = torch.tensor(mask)
        mask = transforms.Resize((800, 1066), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask = transforms.Pad((11, 0))(mask)    
       
        scale = 1066 / original_width 
        bbox = torch.tensor(bbox)
        bbox *= scale
        bbox[:, [0, 2]] += 11  
        # check flag

        assert img.shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.shape[0]
        return img, mask, bbox


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        # TODO: collect_fn
        images, labels, masks, bounding_boxes = list(zip(*batch))
        # Stack images into a single tensor
        images = torch.stack(images)
        # Labels, masks, and bounding boxes remain as lists of tensors
        return images, labels, masks, bounding_boxes

    def loader(self):
        # TODO: return a dataloader
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers, 
            collate_fn=self.collect_fn
        )

## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(0)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 4
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_color_mapping = {
        1: (1, 0, 0),  # Red for class 1
        2: (0, 1, 0),  # Green for class 2
        3: (0, 0, 1)   # Blue for class 3
    }
    for iter, data in enumerate(train_loader, 0):

        img, label, mask, bbox = [data[i] for i in range(len(data))]
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]

        fig, axes = plt.subplots(1, batch_size, figsize=(12 * batch_size, 8))
        # plot the origin img
        for i in range(batch_size):

            # Unnormalize the image for plotting
            img_np = img[i].cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
            img_np = img_np * std + mean
            img_np = np.transpose(img_np, (1,2,0))
            img_np = np.clip(img_np, 0, 1)  # Ensure the values are between 0 and 1

            ax = axes[i]
            ax.imshow(img_np)
            ax.axis('off')

            n_obj = mask[i].shape[0]  # Number of objects in the image

            for j in range(n_obj):
                # Get the mask, bounding box, and label for each object
                mask_j = mask[i][j].cpu().numpy()
                bbox_j = bbox[i][j].cpu().numpy()
                label_j = label[i][j].item()

                # Create an RGBA mask where the alpha channel is based on the mask
                color = class_color_mapping.get(label_j, (1, 1, 0))  # Default to yellow if label not found
                colored_mask = np.zeros((mask_j.shape[0], mask_j.shape[1], 4))
                colored_mask[..., :3] = color  # RGB channels
                colored_mask[..., 3] = mask_j * 0.5  # Alpha channel

                # Overlay the mask with transparency
                ax.imshow(colored_mask)

                # Draw the bounding box
                x1, y1, x2, y2 = bbox_j
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                # Annotate the label
                ax.text(x1, y1 - 10, f'Class {label_j}', fontsize=12, color='white',
                        bbox=dict(facecolor='black', alpha=0.5))

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Save and display the figure
        plt.savefig(f"./testfig/visualtrainset_batch_{iter}.png")
        plt.show()
        plt.close(fig)

        if iter == 5:
            break