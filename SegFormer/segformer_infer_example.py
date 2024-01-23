"""
https://github.com/huggingface/transformers/blob/main/src/transformers/models/segformer/modeling_segformer.py
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Fine_tune_SegFormer_on_custom_dataset.ipynb
https://huggingface.co/blog/fine-tune-segformer
"""
import requests, zipfile, io
# from datasets import load_dataset

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation
import json
from huggingface_hub import cached_download, hf_hub_url
import evaluate
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def download_data():
    url = "https://www.dropbox.com/s/l1e45oht447053f/ADE20k_toy_dataset.zip?dl=1"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train

        sub_path = "training" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "annotations", sub_path)
        
        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)
        
        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
          annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs


def main():
    load_entire_dataset = False

    # if load_entire_dataset:
    #     dataset = load_dataset("scene_parse_150")
    
    root_dir = '/home/qiyuan/2024spring/SegFormer-Pytorch/ADE20k_toy_dataset'
    feature_extractor = SegformerFeatureExtractor(reduce_labels=True)

    train_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor)
    valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor, train=False)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(valid_dataset))
    encoded_inputs = train_dataset[0]
    print(encoded_inputs["pixel_values"].shape)
    print(encoded_inputs["labels"].shape)
    print(encoded_inputs["labels"])
    print(encoded_inputs["labels"].squeeze().unique())
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2)
    # load id2label mapping from a JSON on the hub
    repo_id = "huggingface/label-files"
    filename = "ade20k-id2label.json"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))

    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # define model
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b1",
                                                            num_labels=150, 
                                                            id2label=id2label, 
                                                            label2id=label2id)
    metric = evaluate.load("mean_iou")
    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(50):  # loop over the dataset multiple times
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # get the inputs;
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            
            loss.backward()
            optimizer.step()

            # evaluate
            with torch.no_grad():
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)
                
                # note that the metric expects predictions + labels as numpy arrays
                metrics = metric.add_batch(predictions=predicted.detach().cpu().numpy(),
                                          references=labels.detach().cpu().numpy())

            # let's print loss and metrics every 100 batches
            if idx % 100 == 0:
                metrics = metric._compute(predictions=predicted.detach().cpu().numpy(),
                                          references=labels.detach().cpu().numpy(),
                    num_labels=len(id2label), 
                                        ignore_index=255,
                                        reduce_labels=False, # we've already reduced the labels before)
                )

                print("Loss:", loss.item())
                print("Mean_iou:", metrics["mean_iou"])
                print("Mean accuracy:", metrics["mean_accuracy"])



if __name__ == '__main__':
    # download_data()  # only run once
    main()     
