import argparse
import datetime
import math
import os
import random

import numpy
import torch
import torchvision as torchvision
from torch import optim
from torchvision import transforms
from resnet import ResNet18, ResBlock, ResNet34
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 16
NUM_EPOCHS = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAL_MAPS = {}
IMG_HW = 64
RGB_C = 3
IMAGE_H_W = (IMG_HW,IMG_HW)

def createResnet18(n_channels, n_classes):
    resnet18 = ResNet18(n_channels, ResBlock, outputs=n_classes)
    resnet18.to(DEVICE)
    return resnet18


def createResnet34(n_channels, n_classes):
    resnet34 = ResNet34(n_channels, ResBlock, outputs=n_classes)
    resnet34.to(DEVICE)
    return resnet34


def load_cifar10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    n_channels = trainset.data[0].shape[-1]

    return train_loader, test_loader, classes, n_channels

def filter_labels(class_idx, labels):
    return labels[:,class_idx].unsqueeze(-1).float()

def load_celebA():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Resize(IMAGE_H_W)])

    is_download = not os.path.exists('./data/celeba')
    trainset = torchvision.datasets.celeba.CelebA(root='./data', split="train", download=is_download, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.celeba.CelebA(root='./data', split="valid", download=is_download, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    classes = ('5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young' )
    n_channels = RGB_C

    return train_loader, test_loader, classes, n_channels

def test_eval(model, loss_fn, reg_loss, class_idx, test_loader):
    running_loss = 0
    half_acc = 0
    total_count = 0
    missed_ones = 0
    missed_zeros = 0
    missed = 0
    contraint_dict = {"smiling":numpy.array([RGB_C, IMG_HW, IMG_HW]), "not_smiling":numpy.array([RGB_C, IMG_HW, IMG_HW])}
    model.front_layer.weight.requires_grad = True

    for i, data in enumerate(test_loader):
        org_img, labels = data
        org_img = org_img.to(DEVICE)
        labels = filter_labels(class_idx, labels).to(DEVICE)

        optimizer.zero_grad()
        flat_img = org_img.reshape(org_img.size(0), org_img.size(1), -1)

        linear_output, outputs = model(flat_img)
        outputs = torch.nn.functional.softmax(outputs, dim=-1)
        loss = loss_fn(outputs.to(torch.float), torch.nn.functional.one_hot(labels.flatten().to(torch.int64), num_classes=2).to(torch.float))

        running_loss += float(loss.item())
        for pred_idx in range(labels.shape[0]):
            pred = torch.max(outputs, dim=-1)[1].item()
            lbl = labels.item()

            total_count += 1
            if (pred >= 0.5 and lbl == 1) or (pred < 0.5 and lbl == 0):
                half_acc += 1
            elif pred >= 0.5 and lbl == 0:
                missed_zeros += 1
                missed += 1
            elif pred < 0.5 and lbl == 1:
                missed_ones += 1
                missed += 1

            #Contraint Mining On Conv Filters
            if i % 1000 == 0:
                update_contraints(contraint_dict, model.front_layer, org_img, outputs, labels, optimizer)

    model.front_layer.weight.requires_grad = False
    print(f'TEST average loss: {running_loss / total_count:.5f}')
    #print(f'TEST positive accuracy: {pos_acc / total_count:.5f}')
    print(f'TEST accuracy: {half_acc / total_count:.5f}')
    print(f'TEST missed shouldve predicted zero: {missed_zeros / missed:.5f}')
    print(f'TEST missed shouldve predicted one: {missed_ones / missed:.5f}')
    running_loss = 0.0
    return half_acc / total_count


def get_grad_map(current_grd, front_layer):
    if current_grd is None:
        current_grd = torch.sum(front_layer.weight.grad, dim=-1)
    else:
        current_grd = torch.sum(front_layer.weight.grad, dim=-1) - current_grd

    # (xi – min(x)) / (max(x) – min(x))
    current_grd = torch.abs(current_grd)
    current_grd = (current_grd - torch.min(current_grd)) / (torch.max(current_grd) - torch.min(current_grd))

    return current_grd


def update_contraints(np_constraint_arr, front_layer, og_img, pred, label, optimizer):
    optimizer.zero_grad()
    l1 = torch.nn.L1Loss()
    current_grd = None

    to_img = transforms.ToPILImage()
    to_bw = torchvision.transforms.Grayscale()

    if torch.argmax(pred, dim=-1).item() == label.item():
        wr_rg = "CORRECT"
    else:
        wr_rg = "MISSED"

    smile_pred = pred.clone()
    oh_lbl = torch.nn.functional.one_hot(label.to(torch.int64)).squeeze(0)

    if label.item() == 1.0:
        smile_pred[0,0] = 0
    else:
        smile_pred[0,1] = 0

    l1_loss = l1(smile_pred, oh_lbl)
    l1_loss.backward(retain_graph=True)

    current_grd = get_grad_map(current_grd, front_layer)

    img_id = random.randrange(100000000,999999999)

    in_grad_img = current_grd.reshape(1, IMAGE_H_W[0], IMAGE_H_W[1])
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

def train(train_loader, optimizer, model, loss_fn, reg_loss, **kwargs):
    classes = kwargs["classes"]
    smiling_idx = classes.index("Smiling")


    to_img = transforms.ToPILImage('RGB')


    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            org_img, labels = data
            org_img = org_img.to(DEVICE)
            labels = filter_labels(smiling_idx, labels).to(DEVICE)

            optimizer.zero_grad()
            flat_img = org_img.reshape(org_img.size(0),org_img.size(1),-1)

            linear_output, outputs = model(flat_img)
            outputs = torch.nn.functional.softmax(outputs, dim=-1)
            loss = loss_fn(outputs.to(torch.float), torch.nn.functional.one_hot(labels.flatten().to(torch.int64), num_classes=2).to(torch.float))
            #loss = loss_fn(outputs.to(torch.float), torch.nn.functional.one_hot(labels.flatten().to(torch.int64), num_classes=2).to(torch.float)) + reg_loss(linear_output, flat_img)
            #loss = reg_loss(linear_output, flat_img)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 0 :
                # to_img(org_img[0]).save(f".\imgs\sum_{hash(org_img[0])}_original.png")
                # to_img(linear_output[0].reshape(RGB_C,IMG_HW,IMG_HW)).save(f".\imgs\sum_{hash(org_img[0])}_translated.png")
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.5f}')
                running_loss = 0.0


        acc = test_eval(model, loss_fn,reg_loss, smiling_idx, test_loader)
        torch.save(model, f"model_epoch-{epoch}_acc-{(acc*100)//1}.pt")


def display_image(image_to_disp):
    img = image_to_disp / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    print(f"Starting Time: {datetime.datetime.now()}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--output_size')
    parser.add_argument('--loss_fn')
    parser.add_argument('--model')
    args = parser.parse_args()

    if args.dataset == "cifar10":
        train_loader, test_loader, classes, n_channels = load_cifar10()
    elif args.dataset == "celebA":
        train_loader, test_loader, classes, n_channels = load_celebA()
    else:
        raise NotImplementedError(f"Dataset not implemented: {args.dataset}")

    kwargs = {"classes":classes, "test_loader":test_loader}
    if args.model == "resnet18":
        model = createResnet18(n_channels, int(args.output_size))
    elif args.model == "resnet34":
        model = createResnet34(n_channels, int(args.output_size))
    else:
        raise NotImplementedError(f"Dataset not implemented: {args.dataset}")



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
    train(train_loader, optimizer, model, loss_fn, reg_loss, **kwargs)
