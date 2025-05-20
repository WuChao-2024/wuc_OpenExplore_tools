def c(model):
    def fps(x):
        return 1/(0.0008583*x + 0.0035866)
    # 计算所有参数的数量
    total_params = sum(p.numel() for p in model.parameters())
    para = total_params / 1e6
    print(f"参数量: {para:.2f} M, {model.__module__} ")


def p(model, indent="", is_last=True, current_depth=0, max_depth=10):
    if current_depth >= max_depth:
        return
    # 获取子模块列表
    children = list(model.named_children())
    total = len(children)
    for i, (name, module) in enumerate(model.named_children()):
        is_last_child = (i == total - 1)
        # 打印当前模块名
        connector = "└──" if is_last_child else "├──"
        # print(f"{father_name}.{name}: [{module.__class__.__name__}]")
        print(f"{indent}{connector}{name}: [{module.__class__.__name__}]")
        # 更新缩进
        new_indent = indent
        if not is_last_child:
            new_indent += "|   "
        else:
            new_indent += "    "
        # 递归打印子模块
        if len(list(module.children())) > 0:
            # try:
            #     int(name)
            #     father_name = father_name + "[" + name + "]"
            # except:
            #     father_name = father_name + "." + name
            p(module, new_indent, is_last=is_last_child, current_depth=current_depth + 1, max_depth=max_depth)

import requests
from PIL import Image
import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

import cv2, os

import numpy as np

import time

device = torch.device("cpu")

image_processor = DepthProImageProcessorFast.from_pretrained("DepthPro_hf")
model = DepthProForDepthEstimation.from_pretrained("DepthPro_hf").to(device)

print(f"model: [{model.__class__.__name__}]")
p(model)
