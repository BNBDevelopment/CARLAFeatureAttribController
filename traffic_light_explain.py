import argparse
import datetime
import os
import random
from pathlib import Path

import PIL
import torch
import torchvision
from torch import optim
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.models import resnet18, resnet50
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 1
NUM_EPOCHS = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAL_MAPS = {}
IMG_HW = 128
RGB_C = 3
IMAGE_H_W = (IMG_HW,IMG_HW)

def filter_labels(class_idx, labels):
    return labels[:,class_idx].unsqueeze(-1).float()

def test_eval(model, loss_fn, reg_loss, class_idx, test_loader):
    running_loss = 0
    half_acc = 0
    total_count = 0
    missed_ones = 0
    missed_zeros = 0
    missed = 0
    model.front_layer.weight.requires_grad = True

    for i, data in enumerate(test_loader):
        org_img, labels = data
        org_img = org_img.to(DEVICE)
        #labels = filter_labels(class_idx, labels).to(DEVICE)

        optimizer.zero_grad()
        flat_img = org_img.reshape(org_img.size(0), org_img.size(1), -1)

        outputs = model(flat_img)
        outputs = torch.nn.functional.softmax(outputs, dim=-1)
        #loss = loss_fn(outputs.to(torch.float), torch.nn.functional.one_hot(labels.flatten().to(torch.int64), num_classes=4).to(torch.float).to(DEVICE))

        #running_loss += float(loss.item())
        for pred_idx in range(labels.shape[0]):
            pred = torch.max(outputs, dim=-1)[1].item()
            lbl = labels.item()
            #
            # total_count += 1
            # if (pred >= 0.5 and lbl == 1) or (pred < 0.5 and lbl == 0):
            #     half_acc += 1
            # elif pred >= 0.5 and lbl == 0:
            #     missed_zeros += 1
            #     missed += 1
            # elif pred < 0.5 and lbl == 1:
            #     missed_ones += 1
            #     missed += 1

            #Contraint Mining On Conv Filters
            #if i % 1000 == 0:
            update_contraints(model.front_layer, org_img, outputs, labels, optimizer)

    #model.front_layer.weight.requires_grad = False
    # print(f'TEST average loss: {running_loss / total_count:.5f}')
    # #print(f'TEST positive accuracy: {pos_acc / total_count:.5f}')
    # print(f'TEST accuracy: {half_acc / total_count:.5f}')
    # print(f'TEST missed shouldve predicted zero: {missed_zeros / missed:.5f}')
    # print(f'TEST missed shouldve predicted one: {missed_ones / missed:.5f}')
    running_loss = 0.0
   # return half_acc / total_count


def get_grad_map(current_grd, front_layer):
    if current_grd is None:
        current_grd = torch.sum(front_layer.weight.grad, dim=-1)
    else:
        current_grd = torch.sum(front_layer.weight.grad, dim=-1) - current_grd

    # (xi – min(x)) / (max(x) – min(x))
    #current_grd = torch.abs(current_grd)
    #current_grd = torch.nn.functional.relu(-1 * current_grd)
    current_grd = 1 * current_grd
    #current_grd = (current_grd - torch.min(current_grd)) / (torch.max(current_grd) - torch.min(current_grd))

    return current_grd


def update_contraints(front_layer, og_img, pred, label, optimizer):
    optimizer.zero_grad()
    l1 = torch.nn.L1Loss()
    current_grd = None

    to_img = transforms.ToPILImage()
    to_bw = torchvision.transforms.Grayscale()

    if torch.argmax(pred, dim=-1).item() == label.item():
        wr_rg = "CORRECT"
    else:
        wr_rg = "MISSED"

    temp_pred = pred.clone()
    oh_lbl = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=4)

    # saved_val = temp_pred[0,label.item()].item()
    # temp_pred = oh_lbl.clone().to(torch.float)
    # temp_pred[label.item()] = saved_val

    new_label = []
    for i in range(oh_lbl.size(-1)):
        if i != label.item():
            new_label.append(pred[0, i].item())
        else:
            new_label.append(1.0)
    new_label = torch.tensor(new_label).unsqueeze(0)
    l1_loss = l1(pred, new_label.to(torch.float).to(DEVICE))
    # l1_loss = l1(pred, oh_lbl.to(torch.float))
    l1_loss.backward(retain_graph=True)

    current_grd = get_grad_map(current_grd, front_layer)

    img_id = random.randrange(100000000,999999999)

    in_grad_img = current_grd.reshape(IMAGE_H_W[0], IMAGE_H_W[1])
    in_grad_img = torch.sum(torch.mul(og_img, in_grad_img), dim=1)
    #in_grad_img = (in_grad_img - torch.min(in_grad_img)) / (torch.max(in_grad_img) - torch.min(in_grad_img))
    #in_grad_img = in_grad_img - torch.mean(in_grad_img)
    to_img(in_grad_img).save(f".\imgs\{img_id}_gradient_{wr_rg}_conf_{pred[:,1].item()}.png")

    orig_img = og_img.reshape(RGB_C, IMAGE_H_W[0], IMAGE_H_W[1])
    to_img(orig_img).save(f".\imgs\{hash(img_id)}_original_{wr_rg}_conf_{pred[:,1].item()}.png")

    bw_orig = to_bw(orig_img)
    mycmap = plt.cm.Reds
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, 259)

    y, x = np.mgrid[0:IMG_HW, 0:IMG_HW]
    fig, ax = plt.subplots(1, 1)
    ax.imshow(bw_orig.cpu().squeeze(0))
    cb = ax.contourf(x, y, in_grad_img.cpu().squeeze(0), 15, cmap=mycmap)
    #plt.colorbar(cb)
    plt.savefig(f".\imgs\{hash(img_id)}_heatmap_{wr_rg}_conf_{pred[:,1].item()}.png", bbox_inches="tight", pad_inches=0)

    optimizer.zero_grad()

def load_local():
    my_transforms = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_x_directory = "./datasets/carla"
    labels = []
    imgs = []
    c_map = {'GREEN':0,'RED':1,'YELLOW':2}
    for filename in os.listdir(train_x_directory):
        img = os.path.join(train_x_directory, filename)

        image_file = PIL.Image.open(img)
        image_t = my_transforms(image_file)
        labels.append(c_map[filename.split()[0]])
        imgs.append(image_t)

    ds = TensorDataset(torch.stack(imgs), torch.as_tensor(labels))
    data_loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE)

    return None, data_loader, None, None


class wrapper_model(torch.nn.Module):

    def __init__(self, loaded_model):
        super().__init__()
        self.front_layer = torch.nn.Linear(IMG_HW*IMG_HW, IMG_HW*IMG_HW).to(DEVICE)
        self.model = loaded_model.to(DEVICE)

    def forward(self,in_data):
        linear_out = self.front_layer(in_data)
        in_data = linear_out.reshape(-1,RGB_C,IMG_HW,IMG_HW)
        final = self.model(in_data)
        return final


if __name__ == '__main__':
    print(f"Starting Time: {datetime.datetime.now()}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_size')
    parser.add_argument('--loss_fn')
    args = parser.parse_args()

    train_loader, test_loader, classes, n_channels = load_local()

    loaded_model = resnet50()
    loaded_model.fc = torch.nn.Linear(2048, 4)
    weights = torch.load(Path("50_finetune.pth"), map_location=torch.device('cuda:0'))
    loaded_model.load_state_dict(weights)

    #loaded_model.eval()

    model = wrapper_model(loaded_model)

    #1 FOR
    color_idx = 1

    #Override initial layer wights so that it is a reconstruction
    with torch.no_grad():
        model.front_layer.weight = torch.nn.Parameter(torch.eye(model.front_layer.weight.size(0)).to(DEVICE))
        model.front_layer.weight.requires_grad = False

    if args.loss_fn == "mse":
        loss_fn = torch.nn.MSELoss()
    elif args.loss_fn == "ce":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif args.loss_fn == "bce":
        loss_fn = torch.nn.BCELoss()
    else:
        raise NotImplementedError(f"Dataset not implemented: {args.dataset}")

    reg_loss = torch.nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    model.eval()
    acc = test_eval(model, loss_fn, reg_loss, color_idx, test_loader)