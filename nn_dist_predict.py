import math
import os
import pickle
from typing import List, Optional, Dict, Tuple, Callable

import torch
import PIL.Image
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.utils.data import TensorDataset
from tqdm import tqdm
from torch.optim import AdamW
from torchvision import datasets, transforms
from torchvision.datasets import Kitti, ImageFolder
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101


EPOCHS = 50
LR = 5e-2
BATCH_SIZE = 16


def create_resnet(type, weights=None):
    if type == 18:
        model = resnet18(weights=weights)
    elif type == 34:
        model = resnet18(weights=weights)
    elif type == 50:
        model = resnet18(weights=weights)
    elif type == 101:
        model = resnet18(weights=weights)
    else:
        raise NotImplementedError(f"Model choice {type} not implemented!")

    model.fc = Linear(in_features=512, out_features=1, bias=True)
    return model
    #resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    #resnet50(weights="IMAGENET1K_V1"

def create_dl_mydataset(root_path="./my_dataset"):
    file_list = os.listdir(root_path)
    img_to_tensor = transforms.ToTensor()
    imgs = []
    lbls = []

    for file in file_list:
        img_file = PIL.Image.open(os.path.join(root_path, file))
        img_t = img_to_tensor(img_file)
        lbl_t = torch.tensor(float(file.split("_")[-1][:-4]))

        imgs.append(img_t)
        lbls.append(lbl_t)

    ds = TensorDataset(torch.stack(imgs), torch.as_tensor(lbls))
    data_loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE)
    return data_loader

def create_dl_kitti(root_path="./datasets"):
    kitti_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(size=(600, 600)),
        transforms.Resize(size=(100, 100))
    ])

    temp = datasets.Kitti(
        root=root_path,
        transform=kitti_transforms,
        download=not os.path.exists(root_path)
    )
    data_loader = torch.utils.data.DataLoader(temp, batch_size=BATCH_SIZE)
    #next(iter(data_loader))

    train_x_directory = "./datasets/Kitti/raw/training/image_2/"
    train_y_directory = "./datasets/Kitti/raw/training/label_2/"
    labels = []
    imgs = []
    for filename in os.listdir(train_x_directory):
        img = os.path.join(train_x_directory, filename)
        lbl = os.path.join(train_y_directory, filename[:-3] + "txt")

        image_file = PIL.Image.open(img)
        image_t = kitti_transforms(image_file)
        with open(lbl, "r") as lbl_file:
            for line in lbl_file:
                if line[0:3] == "Car":
                    imgs.append(image_t)
                    distances = line.split()[11:14]
                    dist = math.sqrt(sum([float(coord) ** 2 for coord in distances]))
                    labels.append(dist)
                    break

    ds = TensorDataset(torch.stack(imgs), torch.as_tensor(labels))
    data_loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE)
    return data_loader


class CarlaDataset(ImageFolder):
    def make_dataset(
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        pass


def create_dl_carla(root_path="./datasets"):
    dataset = CarlaDataset(
        root=root_path,
        transform=transforms.ToTensor()
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    return data_loader


def save_model(model, path='model.pt'):
    with open(path, 'wb') as file:
        model_dict = {"model": model}
        pickle.dump(model_dict, file)


def load_model(path='model.pt', device='cpu'):
    with open(path, 'rb') as file:
        model_dict = pickle.load(file)
        return model_dict["model"].to(device), model_dict["seq_len"]


def train(model, dataloader):
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = MSELoss() #CrossEntropyLoss() #MSELoss


    for epoch in range(EPOCHS):
        progress_bar = tqdm(range(BATCH_SIZE, len(dataloader), BATCH_SIZE), position=0, leave=True)
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_loss = 0

        for ids in dataloader:
            input_ids = ids[0]
            labels = ids[1]
            #, y = labels
            outputs = model(x=input_ids)
            loss = loss_fn(outputs.to(torch.float), labels.to(torch.float))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
            epoch_loss += loss.item()

        print(f"\nEpoch {epoch} average loss is: {epoch_loss/((len(dataloader) / BATCH_SIZE))}")

        if (epoch+1) % 10 == 0 or epoch == EPOCHS:
            save_model(model, path=f'model_epoch_{epoch}.pt')

    return model


if __name__ == "__main__":
    model = create_resnet(50)

    dataloader_kitti = create_dl_kitti()
    train(model, dataloader_kitti)

    dataloader_mydataset = create_dl_mydataset()
    train(model, dataloader_mydataset)