from matplotlib.transforms import Transform
import torch
from torch import nn
from torch import optim

from inclearn import models
from inclearn.convnet import resnet, cifar_resnet, modified_resnet_cifar, preact_resnet, var_cnn,varcnn
from inclearn.datasets import data
from inclearn.convnet.resnet import SEFeatureAt
from inclearn.convnet.var_cnn import SEFeatureAt1D

def get_optimizer(params, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise NotImplementedError

def get_attention(inplane, type, at_res):
    return SEFeatureAt1D(inplane, type, at_res)

def get_convnet(convnet_type, **kwargs):
    if convnet_type == "resnet18":
        return resnet.resnet18(**kwargs)
    elif convnet_type == "resnet32":
        return cifar_resnet.resnet32()
    elif convnet_type == "var_cnn":
        return var_cnn.ResNet18(**kwargs)
    elif convnet_type == "varcnn":
        return varcnn.ResNet18(**kwargs)
    else:
        raise NotImplementedError("Unknwon convnet type {}.".format(convnet_type))


def get_model(cfg, trial_i, _run, ex, tensorboard, inc_dataset):
    if cfg["model"] == "incmodel":
        return models.IncModel(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    else:
        raise NotImplementedError(cfg["model"])


def get_data(cfg, trial_i):
    return data.IncrementalDataset(
        trial_i=trial_i,
        dataset_name=cfg["dataset"],
        random_order=cfg["random_classes"],
        shuffle=True,
        batch_size=cfg["batch_size"],
        workers=cfg["workers"],
        validation_split=cfg["validation"],
        resampling=cfg["resampling"],
        increment=cfg["increment"],
        start_class=cfg["start_class"],
    )


def set_device(cfg):
    device_type = cfg["device"]

    if device_type == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(device_type))

    cfg["device"] = device
    return device


def set_device(cfg):
    device_type = cfg["device"]
    available_gpus = torch.cuda.device_count()
    print(f"Debug: Available GPUs {available_gpus}")  

    device_type = cfg["device"]

    if device_type == -1:
        device = torch.device("cpu")
    elif 0 <= int(device_type) < available_gpus:
        device = torch.device(f"cuda:{device_type}")
    else:
        raise RuntimeError(f"CUDA device index {device_type} not available. Available GPUs: {available_gpus}")


    cfg["device"] = device
    return device