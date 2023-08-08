################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################
import sys
import os

# Add the current directory to PYTHONPATH for YoloV7
sys.path.insert(0, os.path.abspath("."))
pydir = os.path.dirname(__file__)

import yaml
import collections
import warnings
import argparse
import json
from pathlib import Path

# PyTorch
import torch
import torch.nn as nn

# YoloV7
import test
from models.yolo import Model
from models.common import Conv
from utils.datasets import create_dataloader
from utils.google_utils import attempt_download
from utils.general import init_seeds

import quantization.quantize as quantize

# Disable all warning
warnings.filterwarnings("ignore")


class SummaryTool:
    def __init__(self, file):
        # Initiate SummaryTool with a given file path to save the summary
        self.file = file
        self.data = []

    def append(self, item):
        # Append an item to the summary and save the current state of the summary into the file
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)

def load_yolov7_model(weight, device) -> Model:
    # Function to load a YoloV7 model with a given weights and device. 
    # The model is evaluated and prepared for inference.

    # Download the model weights if not already available
    attempt_download(weight)
    # Load the model weights
    model = torch.load(weight, map_location=device)["model"]
    # Loop through each module in the model to ensure compatibility with specific PyTorch versions
    for m in model.modules():
        if type(m) is nn.Upsample:
            # Ensure compatibility with PyTorch 1.11.0
            m.recompute_scale_factor = None
        elif type(m) is Conv:
            # Ensure compatibility with PyTorch 1.6.0
            m._non_persistent_buffers_set = set() 

    model.float() # Convert the model to float datatype
    model.eval() # Set the model to evaluation mode

    with torch.no_grad(): 
        model.fuse() # Fuse the model layers
    return model

# Similar functions for loading custom training and validation data using a given directory, txt file and other parameters
# Hyperspectral parameters are loaded from a yaml file
def create_custom_train_dataloader(datadir, train_txt_filename="train.txt", batch_size=4, image_size=640, single_cls=False, rect=False, image_weights=False):
    with open("data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    loader = create_dataloader(
        f"{datadir}/"+train_txt_filename, 
        imgsz=image_size, 
        batch_size=batch_size, 
        opt=collections.namedtuple("Opt", "single_cls")(single_cls),
        augment=True, hyp=hyp, rect=rect, cache=False, stride=32, pad=0, image_weights=image_weights)[0]
    return loader

def create_custom_val_dataloader(datadir, val_txt_filename="val.txt", batch_size=4, keep_images=None, image_size=640, single_cls=False, rect=True, image_weights=False):

    loader = create_dataloader(
        f"{datadir}/"+val_txt_filename, 
        imgsz=image_size, 
        batch_size=batch_size, 
        opt=collections.namedtuple("Opt", "single_cls")(single_cls),
        augment=False, hyp=None, rect=rect, cache=False,stride=32,pad=0.5, image_weights=image_weights)[0]

    def subclass_len(self):
        if keep_images is not None:
            return keep_images
        return len(self.img_files)

    loader.dataset.__len__ = subclass_len
    return loader

# Function to evaluate the performance of a given model on custom data
def evaluate_custom(model, dataloader, using_cocotools = False, save_dir=".", conf_thres=0.001, iou_thres=0.65):
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    return test.test(
        "data/custom_dataset.yaml", 
        save_dir=Path(save_dir),
        dataloader=dataloader, conf_thres=conf_thres,iou_thres=iou_thres,model=model,is_coco=True,
        plots=False,half_precision=True,save_json=using_cocotools)[0][3]

# Additional functions for loading and evaluating the model on COCO dataset 
def create_coco_train_dataloader(cocodir, batch_size=10):
    with open("data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    loader = create_dataloader(
        f"{cocodir}/train2017.txt", 
        imgsz=640, 
        batch_size=batch_size, 
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=True, hyp=hyp, rect=False, cache=False, stride=32,pad=0, image_weights=False)[0]
    return loader

# Additional functions for exporting the model in ONNX format
def export_onnx(model : Model, file, size=640, dynamic_batch=False):
    device = next(model.parameters()).device
    model.float()
    dummy = torch.zeros(1, 3, size, size, device=device)
    model.model[-1].concat = True
    grid_old_func = model.model[-1]._make_grid
    model.model[-1]._make_grid = lambda *args: torch.from_numpy(grid_old_func(*args).data.numpy())
    quantize.export_onnx(model, dummy, file, opset_version=13, 
        input_names=["images"], output_names=["outputs"], 
        dynamic_axes={"images": {0: "batch"}, "outputs": {0: "batch"}} if dynamic_batch else None
    )
    model.model[-1].concat = False
    model.model[-1]._make_grid = grid_old_func


# The main function for model quantization
def cmd_quantize(weight, custom, datadir, train_txt_filename, val_txt_filename, image_size, single_cls, rect, image_weights, device, ignore_policy, save_ptq, save_qat, supervision_stride, iters, eval_origin, eval_ptq):
    
    # Initialize the quantization process
    quantize.initialize()

    # Create directories for saving the Post-training Quantization (PTQ) model, if provided and non-empty
    if save_ptq and os.path.dirname(save_ptq) != "":
        os.makedirs(os.path.dirname(save_ptq), exist_ok=True)

    # Create directories for saving the Quantization-Aware Training (QAT) model, if provided and non-empty
    if save_qat and os.path.dirname(save_qat) != "":
        os.makedirs(os.path.dirname(save_qat), exist_ok=True)

    # Set the device for the model training/quantization
    device  = torch.device(device)
    # Load YOLOv7 model with weights
    model   = load_yolov7_model(weight, device)
    # If the model is custom, create custom dataloaders for both training and validation data
    # Else, create default dataloaders for COCO dataset
    if custom:
        train_dataloader = create_custom_train_dataloader(
            datadir,
            train_txt_filename=train_txt_filename,
            image_size=image_size,
            single_cls=single_cls,
            rect=rect,
            image_weights=image_weights,
        )
        val_dataloader   = create_custom_val_dataloader(
            datadir,
            val_txt_filename=val_txt_filename,
            image_size=image_size,
            single_cls=single_cls,
            rect=True,
            image_weights=False)
    else:
        train_dataloader = create_coco_train_dataloader(datadir)
        val_dataloader   = create_coco_val_dataloader(datadir)

    # Replace original modules in model with quantizable modules 
    quantize.replace_to_quantization_module(model, ignore_policy=ignore_policy)
    # Apply custom rules to the quantizer
    quantize.apply_custom_rules_to_quantizer(model, export_onnx)
    # Calibrate the model for quantization using training data
    quantize.calibrate_model(model, train_dataloader, device)

    # Prepare to save the quantization summary to a JSON file
    json_save_dir = "." if os.path.dirname(save_ptq) == "" else os.path.dirname(save_ptq)
    summary_file = os.path.join(json_save_dir, "summary.json")
    summary = SummaryTool(summary_file)

    # Evaluate the original model and append results to summary
    if eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            if custom:
                ap = evaluate_custom(model, val_dataloader, True, json_save_dir)
            else:
                ap = evaluate_coco(model, val_dataloader, True, json_save_dir)
            summary.append(["Origin", ap])

    # Evaluate the PTQ model and append results to summary
    if eval_ptq:
        print("Evaluate PTQ...")
        if custom:
            ap = evaluate_custom(model, val_dataloader, True, json_save_dir)
        else:
            ap = evaluate_coco(model, val_dataloader, True, json_save_dir)
        summary.append(["PTQ", ap])

    # Save the PTQ model if required
    if save_ptq:
        print(f"Save ptq model to {save_ptq}")
        torch.save({"model": model}, save_ptq)

    # If there's no need to save QAT model, terminate the function
    if save_qat is None:
        print("Done as save_qat is None.")
        return

    best_ap = 0

    # Function to be run for each epoch during QAT
    def per_epoch(model, epoch, lr):

        nonlocal best_ap
        # Evaluate the QAT model
        if custom:
            ap = evaluate_custom(model, val_dataloader, True, json_save_dir)
        else:
            ap = evaluate_coco(model, val_dataloader, True, json_save_dir)
        summary.append([f"QAT{epoch}", ap])

        # If the model's average precision is the best so far, save it
        if ap > best_ap:
            print(f"Save qat model to {save_qat} @ {ap:.5f}")
            best_ap = ap
            torch.save({"model": model}, save_qat)

    # Preprocessing function for data 
    def preprocess(datas):
        return datas[0].to(device).float() / 255.0

    # Define the supervision policy for QAT
    def supervision_policy():
        supervision_list = []
        for item in model.model:
            supervision_list.append(id(item))

        # Prepare the indices of layers to be supervised during QAT
        keep_idx = list(range(0, len(model.model) - 1, supervision_stride))
        keep_idx.append(len(model.model) - 2)
        def impl(name, module):
            if id(module) not in supervision_list: return False
            idx = supervision_list.index(id(module))
            # If the layer is supervised, print a statement
            if idx in keep_idx:
                print(f"Supervision: {name} will compute loss with origin model during QAT training")
            else:
                print(f"Supervision: {name} no compute loss during QAT training, that is unsupervised only and doesn't mean don't learn")
            return idx in keep_idx
        return impl

    # Perform QAT fine-tuning on the model
    quantize.finetune(
        model, train_dataloader, per_epoch, early_exit_batchs_per_epoch=iters, 
        preprocess=preprocess, supervision_policy=supervision_policy())

# Function to export a PyTorch model to ONNX format
def cmd_export(weight, save, size, dynamic):
    
    # Initialize the quantization process
    quantize.initialize()
    # If save path is not provided, set it by default to be the same as the input model file, but with a '.onnx' extension
    if save is None:
        name = os.path.basename(weight)
        name = name[:name.rfind('.')]
        save = os.path.join(os.path.dirname(weight), name + ".onnx")
        
    # Export model to ONNX
    export_onnx(torch.load(weight, map_location="cpu")["model"], save, size, dynamic_batch=dynamic)
    print(f"Save onnx to {save}")

# Function for sensitivity analysis of the model
def cmd_sensitive_analysis(weight, device, cocodir, summary_save, num_image):

    # Initialize the quantization process
    quantize.initialize()
    # Set the device for the analysis
    device  = torch.device(device)
    # Load the YOLOv7 model
    model   = load_yolov7_model(weight, device)
    # Create dataloaders for COCO dataset
    train_dataloader = create_coco_train_dataloader(cocodir)
    val_dataloader   = create_coco_val_dataloader(cocodir, keep_images=None if num_image is None or num_image < 1 else num_image)
    # Replace original modules in the model with quantizable ones
    quantize.replace_to_quantization_module(model)
    # Calibrate the model for quantization
    quantize.calibrate_model(model, train_dataloader)

    # Initialize a tool for saving the analysis summary
    summary = SummaryTool(summary_save)
    print("Evaluate PTQ...")
    # Evaluate PTQ model and append results to summary
    ap = evaluate_coco(model, val_dataloader)
    summary.append([ap, "PTQ"])

    print("Sensitive analysis by each layer...")
    # Evaluate the model with each layer quantized separately
    for i in range(0, len(model.model)):
        layer = model.model[i]
        # If the layer has a quantizer, disable it, evaluate the model, enable it back and append results to summary
        if quantize.have_quantizer(layer):
            print(f"Quantization disable model.{i}")
            quantize.disable_quantization(layer).apply()
            ap = evaluate_coco(model, val_dataloader)
            summary.append([ap, f"model.{i}"])
            quantize.enable_quantization(layer).apply()
        else:
            print(f"ignore model.{i} because it is {type(layer)}")

    # Sort the summary by model's average precision in descending order
    summary = sorted(summary.data, key=lambda x:x[0], reverse=True)
    print("Sensitive summary:")
    # Print the top 10 results
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")


# Define function to test the model
def cmd_test(weight, device, cocodir, confidence, nmsthres):
    device  = torch.device(device)  # Define the device
    model   = load_yolov7_model(weight, device)  # Load the YOLOv7 model
    val_dataloader   = create_coco_val_dataloader(cocodir)  # Create the validation dataloader
    # Evaluate the model on the COCO dataset
    evaluate_coco(model, val_dataloader, True, conf_thres=confidence, iou_thres=nmsthres)

# Define the main function
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='qat.py')
    subps  = parser.add_subparsers(dest="cmd")
    exp    = subps.add_parser("export", help="Export weight to onnx file")
    exp.add_argument("weight", type=str, default="yolov7.pt", help="export pt file")
    exp.add_argument("--save", type=str, required=False, help="export onnx file")
    exp.add_argument("--size", type=int, default=1056, help="export input size")
    exp.add_argument("--dynamic", action="store_true", help="export dynamic batch")

    qat = subps.add_parser("quantize", help="PTQ/QAT finetune ...")
    qat.add_argument("weight", type=str, nargs="?", default="yolov7.pt", help="weight file")
    qat.add_argument("--custom", action="store_true", help="custom dataset and parameters")
    qat.add_argument("--datadir", type=str, default="/datav/dataset/coco", help="data directory")
    qat.add_argument("--train-file", type=str, default="train.txt", help="train set txt file name (eg train.txt)")
    qat.add_argument("--val-file", type=str, default="val.txt", help="validation set txt file name (eg val.txt)")
    qat.add_argument("--image-size", type=int, default=1056, help="image size")
    qat.add_argument("--single-cls", action="store_true", help="single class")
    qat.add_argument("--rect", action="store_true", help="Rectangular training")
    qat.add_argument("--img-wts", action="store_true", help="Image weights")
    qat.add_argument("--device", type=str, default="cuda:0", help="device")
    qat.add_argument("--ignore-policy", type=str, default="model\.105\.m\.(.*)", help="regx")
    qat.add_argument("--ptq", type=str, default="ptq.pt", help="file")
    qat.add_argument("--qat", type=str, default=None, help="file")
    qat.add_argument("--supervision-stride", type=int, default=1, help="supervision stride")
    qat.add_argument("--iters", type=int, default=200, help="iters per epoch")
    qat.add_argument("--eval-origin", action="store_true", help="do eval for origin model")
    qat.add_argument("--eval-ptq", action="store_true", help="do eval for ptq model")

    sensitive = subps.add_parser("sensitive", help="Sensitive layer analysis")
    sensitive.add_argument("weight", type=str, nargs="?", default="yolov7.pt", help="weight file")
    sensitive.add_argument("--device", type=str, default="cuda:0", help="device")
    sensitive.add_argument("--cocodir", type=str, default="/datav/dataset/coco", help="coco directory")
    sensitive.add_argument("--summary", type=str, default="sensitive-summary.json", help="summary save file")
    sensitive.add_argument("--num-image", type=int, default=None, help="number of image to evaluate")

    testcmd = subps.add_parser("test", help="Do evaluate")
    testcmd.add_argument("weight", type=str, default="yolov7.pt", help="weight file")
    testcmd.add_argument("--cocodir", type=str, default="/datav/dataset/coco", help="coco directory")
    testcmd.add_argument("--device", type=str, default="cuda:0", help="device")
    testcmd.add_argument("--confidence", type=float, default=0.001, help="confidence threshold")
    testcmd.add_argument("--nmsthres", type=float, default=0.65, help="nms threshold")

    args = parser.parse_args()  # Parse the arguments
    init_seeds(57)  # Initialize seeds

    # Execute the corresponding function based on the parsed command
    print("\n Arguments: \n", args)
    if args.cmd == "export":
        # If the command is 'export', the script calls the 'cmd_export' function.
        # The function is passed parameters for the weights file, the output file to save to, the image size, and whether to export dynamically.
        # The 'export' command is intended to convert the PyTorch model into an ONNX (Open Neural Network Exchange) model.
        cmd_export(args.weight, args.save, args.size, args.dynamic)

    elif args.cmd == "quantize":
        # If the command is 'quantize', the script calls the 'cmd_quantize' function.
        # It prints the provided arguments and then executes the 'quantize' function, which applies Post-training Quantization (PTQ) or 
        # Quantization-Aware Training (QAT) to the model.
        # The purpose of PTQ and QAT is to reduce the computational resources required by the model, which can be critical for deploying models to devices
        # with limited resources, such as mobile devices.

        cmd_quantize(
            weight=args.weight, custom=args.custom, datadir=args.datadir, train_txt_filename=args.train_file, 
            val_txt_filename=args.val_file, image_size=args.image_size, single_cls=args.single_cls,
            rect=args.rect, image_weights=args.img_wts, device=args.device, ignore_policy=args.ignore_policy, 
            save_ptq=args.ptq, save_qat=args.qat, supervision_stride=args.supervision_stride, iters=args.iters,
            eval_origin=args.eval_origin, eval_ptq=args.eval_ptq,
        )
    elif args.cmd == "sensitive":
        # If the 'sensitive' command is called, the script performs a sensitivity analysis of the model layers.
        # It takes the path to the model file, the device to use for computations, the path to the COCO dataset,
        # the output path for the summary file, and the number of images to use in the evaluation.
        cmd_sensitive_analysis(args.weight, args.device, args.cocodir, args.summary, args.num_image)
        
    elif args.cmd == "test":
        # If the 'test' command is called, the script evaluates the model on the COCO dataset.
        # It takes the path to the model file, the device to use for computations, the path to the COCO dataset,
        # the confidence threshold for detection, and the NMS threshold for post-processing.
        cmd_test(args.weight, args.device, args.cocodir, args.confidence, args.nmsthres)

    else:
        # If no valid command is parsed from the command-line arguments, the script will print the help message,
        # providing information about the available commands and their respective options and arguments.
        parser.print_help()