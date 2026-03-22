import os
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from monai.networks.nets import resnet18
from monai.visualize import GradCAM
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    Resized,
    EnsureTyped
)

# ---------------------------------------------------------
# 1. SETUP & LOAD MODEL
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=len(target_cols)).to(device)
model.load_state_dict(torch.load("best_resnet3d_8gb_vram.pth", map_location=device))
model.eval() 

# ---------------------------------------------------------
# 2. INITIALIZE GRAD-CAM
# ---------------------------------------------------------
# Using layer4 shows the deepest, most "blob-like" vision of the network
cam_extractor = GradCAM(nn_module=model, target_layers="layer4") 

# ---------------------------------------------------------
# 3. PREPARE THE 96x96x96 TEST IMAGE
# ---------------------------------------------------------
test_image_path = r"D:\Recuperacion\Escuela\8vo Semestre\ProyectoIntegrador1\MBH_Train_2025_case-label\ID_0a01a6c4_ID_f361ca3063.nii.gz"

val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear")),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image"], source_key="image"),
    Resized(keys=["image"], spatial_size=(96, 96, 96)), # THIS IS WHAT THE NETWORK SEES
    EnsureTyped(keys=["image"], device=device)
])

processed_data = val_transforms({"image": test_image_path})
input_tensor = processed_data["image"].unsqueeze(0)

# ---------------------------------------------------------
# 4. GENERATE HEATMAP
# ---------------------------------------------------------
class_index_to_visualize = 4 
print(f"Generating RAW 96x96x96 CAM for {target_cols[class_index_to_visualize]}...")

heatmap = cam_extractor(x=input_tensor, class_idx=class_index_to_visualize)

# EXTRACT THE EXACT 96x96x96 TENSORS
img_np = input_tensor.squeeze().cpu().detach().numpy()     # The pixelated brain
heatmap_np = heatmap.squeeze().cpu().detach().numpy()      # The pixelated heatmap

# Normalize heatmap
heatmap_np = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np) + 1e-8)

THRESHOLD = 0.5
binary_mask = (heatmap_np < THRESHOLD).astype(np.uint8)

# ---------------------------------------------------------
# 5. INTERACTIVE MATPLOTLIB VISUALIZATION (LOW-RES)
# ---------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(bottom=0.2) 

# Max Z is now exactly 95 (since shape is 96x96x96)
max_z = img_np.shape[2] - 1
initial_z = np.argmax(np.max(heatmap_np, axis=(0, 1))) 

def get_rotated_slice(volume, z_idx):
    return np.rot90(volume[:, :, z_idx])

# Panel 1: The Raw 96x96x96 Input
img_display = ax1.imshow(get_rotated_slice(img_np, initial_z), cmap="gray")
ax1.set_title("Raw ResNet Input (96x96x96)")
ax1.axis("off")

# Panel 2: The Raw Heatmap Overlay
cam_base = ax2.imshow(get_rotated_slice(img_np, initial_z), cmap="gray")
cam_data = get_rotated_slice(heatmap_np, initial_z)
cam_overlay = ax2.imshow(np.ma.masked_where(cam_data < 0.2, cam_data), cmap="jet", alpha=0.5, vmin=0, vmax=1)
ax2.set_title(f"Raw Grad-CAM (Layer 4)")
ax2.axis("off")

# Panel 3: The Raw Threshold Mask
mask_base = ax3.imshow(get_rotated_slice(img_np, initial_z), cmap="gray")
mask_data = get_rotated_slice(binary_mask, initial_z)
mask_overlay = ax3.imshow(np.ma.masked_where(mask_data == 0, mask_data), cmap="autumn", alpha=0.6, vmin=0, vmax=1)
ax3.set_title(f"Raw Mask (Threshold > {THRESHOLD})")
ax3.axis("off")

# Add the slider
ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
slider = Slider(
    ax=ax_slider,
    label='Z-Axis Slice',
    valmin=0,
    valmax=max_z,
    valinit=initial_z,
    valstep=1
)

def update(val):
    z = int(slider.val)
    
    img_display.set_data(get_rotated_slice(img_np, z))
    
    cam_base.set_data(get_rotated_slice(img_np, z))
    new_cam = get_rotated_slice(heatmap_np, z)
    cam_overlay.set_data(np.ma.masked_where(new_cam < 0.2, new_cam))
    
    mask_base.set_data(get_rotated_slice(img_np, z))
    new_mask = get_rotated_slice(binary_mask, z)
    mask_overlay.set_data(np.ma.masked_where(new_mask == 0, new_mask))
    
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()


# import os
# import torch
# import numpy as np
# import nibabel as nib
# import SimpleITK as sitk
# from monai.networks.nets import resnet18
# from monai.visualize import GradCAM
# from monai.transforms import (
#     Compose,
#     LoadImaged,
#     EnsureChannelFirstd,
#     Spacingd,
#     ScaleIntensityRangePercentilesd,
#     CropForegroundd,
#     Resized,
#     EnsureTyped
# )

# # ---------------------------------------------------------
# # 1. SETUP & LOAD MODEL
# # ---------------------------------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# target_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

# model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=len(target_cols)).to(device)
# model.load_state_dict(torch.load("best_resnet3d_8gb_vram.pth", map_location=device))
# model.eval() 

# # ---------------------------------------------------------
# # 2. INITIALIZE GRAD-CAM (CHANGED TO LAYER 1 FOR HIGH RESOLUTION)
# # ---------------------------------------------------------
# # layer1 provides a 48x48x48 feature map instead of 6x6x6
# cam_extractor = GradCAM(nn_module=model, target_layers="layer1") 

# # ---------------------------------------------------------
# # 3. PREPARE A TEST IMAGE
# # ---------------------------------------------------------
# test_image_path = r"D:\Recuperacion\Escuela\8vo Semestre\ProyectoIntegrador1\MBH_Train_2025_case-label\ID_0a01a6c4_ID_f361ca3063.nii.gz"

# val_transforms = Compose([
#     LoadImaged(keys=["image"]),
#     EnsureChannelFirstd(keys=["image"]),
#     Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear")),
#     ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
#     CropForegroundd(keys=["image"], source_key="image"),
#     Resized(keys=["image"], spatial_size=(96, 96, 96)),
#     EnsureTyped(keys=["image"], device=device)
# ])

# processed_data = val_transforms({"image": test_image_path})
# input_tensor = processed_data["image"].unsqueeze(0)

# # ---------------------------------------------------------
# # 4. GENERATE HEATMAP
# # ---------------------------------------------------------
# class_index_to_visualize = 4 
# print(f"Generating high-res CAM for {target_cols[class_index_to_visualize]}...")

# heatmap = cam_extractor(x=input_tensor, class_idx=class_index_to_visualize)
# heatmap_np = heatmap.squeeze().cpu().detach().numpy()
# heatmap_np = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np) + 1e-8)

# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider
# import torch.nn.functional as F

# print("Preparing interactive visualization...")

# # ---------------------------------------------------------
# # 5. PREPARE HIGH-RES ALIGNED IMAGE
# # ---------------------------------------------------------
# hr_transforms = Compose([
#     LoadImaged(keys=["image"]),
#     EnsureChannelFirstd(keys=["image"]),
#     Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear")),
#     ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
#     CropForegroundd(keys=["image"], source_key="image")
# ])

# hr_data = hr_transforms({"image": test_image_path})
# hr_img_np = hr_data["image"].squeeze().numpy()  # Shape: [X, Y, Z]

# # ---------------------------------------------------------
# # 6. UPSCALE CAM AND APPLY THRESHOLD
# # ---------------------------------------------------------
# heatmap_tensor = torch.tensor(heatmap_np).unsqueeze(0).unsqueeze(0) 
# hr_shape = hr_img_np.shape 

# hr_heatmap_tensor = F.interpolate(heatmap_tensor, size=hr_shape, mode='trilinear', align_corners=False)
# hr_heatmap_np = hr_heatmap_tensor.squeeze().numpy()
# hr_heatmap_np = (hr_heatmap_np - hr_heatmap_np.min()) / (hr_heatmap_np.max() - hr_heatmap_np.min() + 1e-8)

# THRESHOLD = 0.5 # Set back to 0.5 so we can see the actual shape
# binary_mask = (hr_heatmap_np > THRESHOLD).astype(np.uint8)

# # ---------------------------------------------------------
# # 7. INTERACTIVE MATPLOTLIB VISUALIZATION WITH SLIDER
# # ---------------------------------------------------------
# # Create figure with space at the bottom for the slider
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
# plt.subplots_adjust(bottom=0.2) # Leave room for slider

# max_z = hr_img_np.shape[2] - 1
# initial_z = np.argmax(np.max(hr_heatmap_np, axis=(0, 1))) # Start at the best slice

# def get_rotated_slice(volume, z_idx):
#     # Extracts the axial slice and rotates it upright
#     return np.rot90(volume[:, :, z_idx])

# # Initial plot
# img_display = ax1.imshow(get_rotated_slice(hr_img_np, initial_z), cmap="gray")
# ax1.set_title("High-Res Original")
# ax1.axis("off")

# cam_base = ax2.imshow(get_rotated_slice(hr_img_np, initial_z), cmap="gray")
# cam_data = get_rotated_slice(hr_heatmap_np, initial_z)
# cam_overlay = ax2.imshow(np.ma.masked_where(cam_data < 0.2, cam_data), cmap="jet", alpha=0.5, vmin=0, vmax=1)
# ax2.set_title(f"Grad-CAM (Layer 1): {target_cols[class_index_to_visualize]}")
# ax2.axis("off")

# mask_base = ax3.imshow(get_rotated_slice(hr_img_np, initial_z), cmap="gray")
# mask_data = get_rotated_slice(binary_mask, initial_z)
# mask_overlay = ax3.imshow(np.ma.masked_where(mask_data == 0, mask_data), cmap="autumn", alpha=0.6, vmin=0, vmax=1)
# ax3.set_title(f"Binary Mask (Threshold > {THRESHOLD})")
# ax3.axis("off")

# # Add the slider
# ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
# slider = Slider(
#     ax=ax_slider,
#     label='Z-Axis Slice',
#     valmin=0,
#     valmax=max_z,
#     valinit=initial_z,
#     valstep=1
# )

# # Update function for when the slider is moved
# def update(val):
#     z = int(slider.val)
    
#     # Update panel 1
#     img_display.set_data(get_rotated_slice(hr_img_np, z))
    
#     # Update panel 2
#     cam_base.set_data(get_rotated_slice(hr_img_np, z))
#     new_cam = get_rotated_slice(hr_heatmap_np, z)
#     cam_overlay.set_data(np.ma.masked_where(new_cam < 0.2, new_cam))
    
#     # Update panel 3
#     mask_base.set_data(get_rotated_slice(hr_img_np, z))
#     new_mask = get_rotated_slice(binary_mask, z)
#     mask_overlay.set_data(np.ma.masked_where(new_mask == 0, new_mask))
    
#     fig.canvas.draw_idle()

# slider.on_changed(update)
# plt.show()
