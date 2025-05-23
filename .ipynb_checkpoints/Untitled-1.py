# %%
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)


# %%
pip install matplotlib rasterio


# %% [markdown]
# # Only run if the session is restarted / the files are cleared !!!

# %%
import zipfile
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

zip_path = "ortho_and_mask.zip"  # or full path like "C:/Users/User/Downloads/tiles.zip"
extract_path = "data"  # where you want to extract

# Create output directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Done unzipping to:", extract_path)


# %%
import rasterio
import matplotlib.pyplot as plt

ortho_path = "data/ortho_clip_rd2.tif"

with rasterio.open(ortho_path) as src:
    img = src.read([1, 2, 3], out_shape=(3, src.height // 8, src.width // 8))

plt.figure(figsize=(12, 8))
plt.imshow(img.transpose(1, 2, 0))
plt.axis('off')
plt.title("Orthophoto Preview")
plt.show()


# %%
import numpy as np
import rasterio
import matplotlib.pyplot as plt

mask_path = "data/LGN2023_AOI_25cm_ok.tif"

with rasterio.open(mask_path) as src:
    downsampled = src.read(
        1,
        out_shape=(src.height // 8, src.width // 8)
    )

plt.figure(figsize=(10, 8))
plt.imshow(downsampled, cmap='nipy_spectral')
plt.axis('off')
plt.title("Downsampled Land Cover Mask Preview")
plt.colorbar()
plt.show()

# %% [markdown]
# # Load Orthophoto

# %%
# Load orthophoto (input image)
from rasterio.windows import Window

window = Window(350, 4300, 6144, 6144)

with rasterio.open(ortho_path) as src_ortho:
    ortho = src_ortho.read([1, 2, 3], window=window)  # Read RGB channels
    ortho_meta = src_ortho.meta

# Load LGN2023 (label mask)
with rasterio.open(mask_path) as src_label:
    label = src_label.read(1, window=window)  # Read single-band labels
    label_meta = src_label.meta

# Normalize ortho image to [0, 1]
ortho = np.moveaxis(ortho, 0, -1)  # Change to H x W x C
ortho = ortho / 255.0

binary_mask = (label == 8).astype(np.uint8)

# Display overlay
plt.figure(figsize=(10, 10))
plt.imshow(ortho)
plt.imshow(binary_mask, cmap='jet', alpha=0)  # Adjust alpha to see overlay
plt.title('Overlay of Ortho Image and LGN Labels')
plt.axis('off')
plt.show()




# %% [markdown]
# # Create dataframe with legend for labeled data of the mask

# %%
import pandas as pd

# Your original LGN class mapping
class_mapping = {
    16: "Water / Canals",
    18: "Buildings",
    19: "Smaller Buildings",
    20: "Light Trees / Forest",
    22: "Forest / Grass",
    23: "Urban Grass",
    28: "Parks / Grass Football Fields",
    251: "Railway & Main Infrastructure",
    252: "Semi-Paved Roads",
    253: "Narrow Roads"
}

# Custom color palette (hex or color names)
class_colors = {
    16: '#253DE8',
    18: '#EC6E21',
    19: '#C4510A',
    20: '#2ECC71',
    22: '#27AE60',
    23: '#0de34e',
    28: '#3B9E06',
    251: '#C6C4CB',
    252: '#F5F385',
    253: '#F5988F'
}

# Create DataFrame
df = pd.DataFrame([
    {
        'value': raw,
        'normalized_value': i + 1,
        'label': class_mapping[raw],
        'color': class_colors[raw]
    }
    for i, raw in enumerate(sorted(class_mapping.keys()))
])

# Save to CSV
df.to_csv("data/lc_data.csv", index=False)

print("✅ Saved")


# %% [markdown]
# # Vizualize labeled mask on top of the ortho_clip

# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

lc_data = pd.read_csv("data/lc_data.csv")

# Extract vals
raw_values = lc_data['value'].tolist()
labels = lc_data['label'].tolist()
colors = lc_data['color'].tolist()

# Create custom colormap
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries=raw_values + [max(raw_values) + 1], ncolors=len(colors))

# Display ortho with label overlay
plt.figure(figsize=(12, 12))
plt.imshow(ortho)
plt.imshow(label, cmap=cmap, norm=norm, alpha=0.7)
plt.title("Overlay: Ortho Image with LGN2023 Labels")
plt.axis('off')

# Build legend
handles = [
    mpatches.Patch(color=colors[i], label=f"{labels[i]} ({raw_values[i]})")
    for i in range(len(raw_values))
]

plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='LGN Classes')
plt.tight_layout()
plt.show()


# %% [markdown]
# # Save window of tifs to new tif files
# Keep in mind that, the ortho is 3-banded (is of type RGB) and that the mask is 1-banded.

# %%
from rasterio.transform import from_origin

with rasterio.open(ortho_path) as src:
    window_transform = src.window_transform(window)
    profile = src.profile.copy()
    profile.update({
        'count': 3,
        'height': ortho.shape[0],
        'width': ortho.shape[1],
        'transform': window_transform
    })

    with rasterio.open("data/ortho_window.tif", "w", **profile) as dst:
        # Reorder bands from HWC to CHW for rasterio
        dst.write(np.moveaxis((ortho * 255).astype('uint8'), -1, 0))

# %%
with rasterio.open(mask_path) as src:
    window_transform = src.window_transform(window)
    profile = src.profile.copy()
    profile.update({
        'count': 1,
        'height': label.shape[0],
        'width': label.shape[1],
        'transform': window_transform,
        'dtype': label.dtype
    })

    with rasterio.open("data/label_window.tif", "w", **profile) as dst:
        dst.write(label, 1)

# %% [markdown]
# # Now that we have the 2 selected areas of interest, we will create the tiles from this data, both for training and testing the model.

# %%
from geotile import GeoTile
import fiona

ortho = "data/ortho_window.tif"

# Step 1: Tile the ortho image
gt_ortho = GeoTile(ortho)
gt_ortho.generate_tiles(output_folder=r'data/tiles_512/ortho', tile_x=512, tile_y=512, stride_x=512, stride_y=512)

# %%
import os

lst = os.listdir("data/tiles_512/ortho") # your directory path
number_files = len(lst)
print("Congrats, you generated", number_files, "new tiles from the ortho image")

# %%
ortho = "data/label_window.tif"

# Step 1: Tile the ortho image
gt_ortho = GeoTile(ortho)
gt_ortho.generate_tiles(output_folder=r'data/tiles_512/label', tile_x=512, tile_y=512, stride_x=512, stride_y=512)

# %%
lst = os.listdir("data/tiles_512/label") # your directory path
number_files = len(lst)
print("Congrats, you generated", number_files, "new tiles from the label image")

# %%
import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Load your class mapping
lc_data = pd.read_csv("data/lc_data.csv")
raw_values = lc_data['value'].tolist()
colors = lc_data['color'].tolist()
labels = lc_data['label'].tolist()
label_dict = dict(zip(raw_values, labels))
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries=raw_values + [max(raw_values) + 1], ncolors=len(colors))

# Paths
ortho_dir = "data/tiles_512/ortho"
label_dir = "data/tiles_512/label"

# Load tile filenames (limit to 20)
tile_files = sorted([f for f in os.listdir(ortho_dir) if f.endswith(".tif")])[:20]

# Plot grid: 4 rows × 5 columns = 20 tiles
n_cols = 5
n_rows = (len(tile_files) + n_cols - 1) // n_cols
plt.figure(figsize=(n_cols * 4, n_rows * 4))

for i, fname in enumerate(tile_files):
    ortho_path = os.path.join(ortho_dir, fname)
    label_path = os.path.join(label_dir, fname)

    # Load ortho
    with rasterio.open(ortho_path) as src_ortho:
        ortho = src_ortho.read([1, 2, 3])
        ortho = np.moveaxis(ortho, 0, -1) / 255.0

    # Load label
    with rasterio.open(label_path) as src_label:
        label = src_label.read(1)

    # Show RGB
    plt.subplot(n_rows * 2, n_cols, i + 1)
    plt.imshow(ortho)
    plt.title(f"Ortho: {fname}")
    plt.axis('off')

    # Show label
    plt.subplot(n_rows * 2, n_cols, i + 1 + len(tile_files))
    plt.imshow(label, cmap=cmap, norm=norm)
    plt.title(f"Label: {fname}")
    plt.axis('on')

plt.tight_layout()
plt.show()


# %% [markdown]
# # Preparing for training

# %%
import tifffile

img = tifffile.imread(r'data\split\train\labels\tile_0_512.tif')
img2 = tifffile.imread(r'data\split\train\ortho\tile_0_512.tif')

print(img.shape)
print(img2.shape)

# %%
import os
import rasterio
import numpy as np

def validate_tile_pair(ortho_path, label_path):
    with rasterio.open(ortho_path) as src:
        ortho = src.read()
    with rasterio.open(label_path) as src:
        label = src.read(1)
    
    assert ortho.shape[1:] == label.shape, f"Size mismatch: {ortho.shape[1:]} vs {label.shape}"
    assert np.any(ortho > 0),   f"Blank ortho tile: {ortho_path}"
    assert np.any(label > 0),   f"Blank label tile: {label_path}"
    return True

ortho_dir = 'data/tiles_512/ortho'
label_dir = 'data/tiles_512/label'

for fname in os.listdir(ortho_dir):
    ortho_path = os.path.join(ortho_dir, fname)
    label_path = os.path.join(label_dir, fname)

    # 3) Wrap in try/except so one bad tile doesn’t stop the loop
    try:
        validate_tile_pair(ortho_path, label_path)
        # print(f"✅  {fname} is valid")
        
    except AssertionError as e:
        print(f"❌  Validation failed for {fname}: {e}")
    except Exception as e:
        print(f"❌  Unexpected error with {fname}: {e}")


# %%
import os
import rasterio
import numpy as np
import pandas as pd
from collections import defaultdict

label_dir = 'data/tiles_512/label'

# 1. Gather all .tif label paths
label_paths = [
    os.path.join(label_dir, f)
    for f in os.listdir(label_dir)
    if f.lower().endswith('.tif')
]

# 2. Count pixels per class
class_counts = defaultdict(int)
for path in label_paths:
    with rasterio.open(path) as src:
        mask = src.read(1)
    unique, counts = np.unique(mask, return_counts=True)
    for cls_value, cnt in zip(unique, counts):
        class_counts[int(cls_value)] += int(cnt)

# 3. Compute class weights for CrossEntropyLoss
total_pixels = sum(class_counts.values())
n_classes = len(class_counts)
class_weights = {
    cls: total_pixels / (cnt * n_classes)
    for cls, cnt in class_counts.items()
}

# 4. Tabulate results
df = pd.DataFrame([
    {'class': cls, 'pixel_count': class_counts[cls], 'weight': class_weights[cls]}
    for cls in sorted(class_counts)
])

print(df.to_string(index=False))


# %% [markdown]
# ## Split

# %%
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import rasterio

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1) Gather all your tile names
ortho_dir = 'data/tiles_512/ortho'
label_dir = 'data/tiles_512/label'
all_files = [f for f in os.listdir(ortho_dir) if f.lower().endswith('.tif')]

# 2) Shuffle & split 80/20
random.seed(42)
random.shuffle(all_files)
split = int(0.8 * len(all_files))
train_files = all_files[:split]
val_files   = all_files[split:]


# %%
import albumentations as A
from albumentations.pytorch import ToTensorV2

mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
train_transform = A.Compose([
    A.RandomResizedCrop(size=(512, 512), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.5),
    A.Affine(translate_percent=0.1, scale=(0.9,1.1), rotate=45, p=0.5),
    A.GaussNoise(p=0.2),
    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])


# %%
import os
import rasterio
import numpy as np
from collections import defaultdict

# 1) Scan your label folder and build class_counts
label_dir = 'data/tiles_512/label'
class_counts = defaultdict(int)
for fname in os.listdir(label_dir):
    if not fname.lower().endswith('.tif'):
        continue
    with rasterio.open(os.path.join(label_dir, fname)) as src:
        mask = src.read(1)
    unique, counts = np.unique(mask, return_counts=True)
    for u, c in zip(unique, counts):
        class_counts[int(u)] += int(c)

# 2) Create orig_classes and mapping
orig_classes = sorted(class_counts.keys())  
# e.g. [16, 18, 19, 20, 22, 23, 28, 251, 252, 253]
mapping = {old: new for new, old in enumerate(orig_classes)}
# Now mapping[16] == 0, mapping[18] == 1, ..., mapping[253] == 9

print("Original classes:", orig_classes)
print("Mapping example:", list(mapping.items())[:5])


# %%
import os
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1) Ensure you have your mapping and transforms defined:
#    class_counts → orig_classes → mapping as before.
#    train_transform, val_transform as before.

class SegmentationDataset(Dataset):
    def __init__(self, files, ortho_dir, label_dir, mapping, transform=None):
        """
        files: list of filenames (e.g. ['tile1.tif', ...])
        ortho_dir, label_dir: paths to your ortho and label folders
        mapping: dict old_label → new_index
        transform: Albumentations Compose for image+mask
        """
        self.files = files
        self.ortho_dir = ortho_dir
        self.label_dir = label_dir
        self.mapping = mapping
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        # load ortho image
        with rasterio.open(os.path.join(self.ortho_dir, f)) as src:
            img = np.transpose(src.read([1,2,3]), (1,2,0))
        # load mask
        with rasterio.open(os.path.join(self.label_dir, f)) as src:
            m = src.read(1)

        # remap labels 16→0, 18→1, etc
        m = np.vectorize(self.mapping.get)(m).astype(np.int64)

        # apply augmentations / normalization
        if self.transform:
            aug = self.transform(image=img, mask=m)
            img, m = aug['image'], aug['mask']
        else:
            img = torch.from_numpy((img.astype(np.float32)/255.0 - 0.5)/0.5).permute(2,0,1)
            m = torch.from_numpy(m)

        return img, m

# 2) Instantiate datasets & loaders
train_ds = SegmentationDataset(
    files=train_files,
    ortho_dir=ortho_dir,
    label_dir=label_dir,
    mapping=mapping,
    transform=train_transform
)
val_ds = SegmentationDataset(
    files=val_files,
    ortho_dir=ortho_dir,
    label_dir=label_dir,
    mapping=mapping,
    transform=val_transform
)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)


# %% [markdown]
# ## UNET

# %%
import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = smp.Unet(
    encoder_name="resnet34",        # pretrained on ImageNet
    encoder_weights="imagenet",
    in_channels=3,
    classes=10,
).to(device)


# %%
# 1) Clamp raw weights

max_w = 10
clamped_weights = {cls: min(w, max_w)
                   for cls, w in class_weights.items()}

# 2) Build the CPU list
weights_list = [clamped_weights[i] for i in sorted(clamped_weights.keys())]

# 3) Create the tensor on CPU, then .to(device)
weights_tensor = torch.tensor(weights_list, dtype=torch.float32)  # CPU
weights_tensor = weights_tensor.to(device)                        # now move to GPU

# 4) Instantiate your loss with these clamped weights
# DiceLoss for multiclass
dice_loss = smp.losses.DiceLoss(mode="multiclass")  
# Keep your weighted CE
ce_loss   = nn.CrossEntropyLoss(weight=weights_tensor)

def loss_fn(preds, masks):
    """
    preds: [B, C, H, W] raw logits
    masks: [B, H, W]   integer labels 0..C-1
    """
    return ce_loss(preds, masks) + dice_loss(preds, masks)

# 5) Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# %% [markdown]
# ## Training

# %%
for epoch in range(1, 100+1):
    # ——— Training ———
    model.train()
    train_loss = 0.0
    train_acc  = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks  = masks.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)      # use loss_fn here
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_acc += pixel_accuracy(preds, masks).item()

    train_loss /= len(train_loader)
    train_acc  /= len(train_loader)

    # ——— Validation ———
    model.eval()
    val_loss = 0.0
    val_acc  = 0.0
    all_ious = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks  = masks.to(device).long()

            outputs = model(images)
            loss = loss_fn(outputs, masks)   # and here
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            val_acc += pixel_accuracy(preds, masks).item()

            _, mean_iou = compute_iou(preds, masks, num_classes)
            all_ious.append(mean_iou)

    val_loss /= len(val_loader)
    val_acc  /= len(val_loader)
    mean_val_iou = torch.stack(all_ious).mean().item()

    print(f"Epoch {epoch:02d}/{num_epochs:02d} "
          f"– Train Loss: {train_loss:.4f} "
          f"Train Acc: {train_acc:.4f} "
          f"– Val Loss: {val_loss:.4f} "
          f"Val Acc: {val_acc:.4f} "
          f"Val mIoU: {mean_val_iou:.4f}")


# %%
import matplotlib.pyplot as plt
import torch

# Make sure your model is in eval mode
model.eval()

# Grab one batch of val data
images, masks = next(iter(val_loader))
images, masks = images.to(device), masks.to(device)

# Run inference
with torch.no_grad():
    outputs = model(images)               # [B, C, H, W]
    preds   = outputs.argmax(dim=1)       # [B, H, W]

# Move everything back to CPU
images = images.cpu()
masks  = masks.cpu()
preds  = preds.cpu()

# Denormalize helper (if you normalized to mean=0.5, std=0.5)
def denormalize(img_tensor):
    img = img_tensor.permute(1,2,0).numpy()   # C×H×W → H×W×C
    img = (img * 0.5) + 0.5                   # unnormalize
    return img.clip(0,1)

# Plot the first N samples
N = 15
fig, axes = plt.subplots(N, 3, figsize=(12, 4*N))
for i in range(N):
    # Ortho image
    ax = axes[i,0]
    ax.imshow(denormalize(images[i]))
    ax.set_title("Ortho")
    ax.axis("off")
    
    # Ground truth
    ax = axes[i,1]
    ax.imshow(masks[i], cmap="tab20", vmin=0, vmax=num_classes-1)
    ax.set_title("Ground Truth")
    ax.axis("off")
    
    # Prediction
    ax = axes[i,2]
    ax.imshow(preds[i], cmap="tab20", vmin=0, vmax=num_classes-1)
    ax.set_title("Prediction")
    ax.axis("off")

plt.tight_layout()
plt.show()


# %%


# %%



