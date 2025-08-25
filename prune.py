import torch
from torchvision.models import resnet18
import torch_pruning as tp

from ultralytics import YOLO
import thop  # for FLOPs computation
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    v10Detect,
    A2C2f,
)
from ultralytics.nn.modules.block import PSABlock, AAttn

# Load the model
# yolo_wrapper = YOLO('/data1/code/competition/tianchi/unseen/yolov12/yolov12l.pt')

# Extract the actual PyTorch model
model = torch.load('/data1/code/competition/tianchi/unseen/yolov12/yolov12l.pt')
model = model['ema' if model.get('ema') else 'model'].float()
# for p in model.parameters():
#     p.requires_grad_(True)
model.info()

# Set up example inputs matching YOLO's expected format
example_inputs = torch.randn(1, 3, 1280, 736)

# Create importance criterion
imp = tp.importance.LAMPImportance(p=2)

# Identify layers to ignore (e.g., detection heads)
ignored_layers = []
for k, m in model.named_modules():
    if isinstance(m, Detect):
        ignored_layers.append(m.cv2[0][2])
        ignored_layers.append(m.cv2[1][2])
        ignored_layers.append(m.cv2[2][2])
        ignored_layers.append(m.cv3[0][2])
        ignored_layers.append(m.cv3[1][2])
        ignored_layers.append(m.cv3[2][2])
        ignored_layers.append(m.dfl)
    if isinstance(m, PSABlock):
        ignored_layers.append(m.attn)
    if isinstance(m, AAttn):
        ignored_layers.append(m)
# breakpoint()
# Initialize pruner with proper configuration
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    iterative_steps=1,
    pruning_ratio=0.1,
    ignored_layers=ignored_layers,
    global_pruning=True,  # Apply global pruning
    root_module_types=[nn.Conv2d, nn.Linear]
    # ch_sparsity=0.5,  # Target channel sparsity
    # root_module_types=[torch.nn.Conv2d],  # Focus on convolutional layers
)

# 3. Prune the model
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
tp.utils.print_tool.before_pruning(model) # or print(model)
pruner.step()
tp.utils.print_tool.after_pruning(pruner.model) # or print(model), this util will show the difference before and after pruning
macs, nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs)
print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")

# 4. finetune the pruned model using your own code.
# finetune(model)
# ...