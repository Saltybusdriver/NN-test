import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
import cv2
import time
import argparse
import sys
from pathlib import Path
import math
import gc
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import torch.nn.functional as F
import pynvml
pynvml_available=True

LEVEL_ERROR = 0
LEVEL_WARN = 1
LEVEL_INFO = 2
LEVEL_DEBUG = 3
current_debug_level = LEVEL_WARN
debug_flag = True


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")


def get_peak_cpu_memory_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        try:
            import psutil
            process = psutil.Process(os.getpid())

            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # print("Warning: Cannot get peak CPU memory. 'resource' or 'psutil' module not found.", file=sys.stderr)
            return -1.0 # 
def get_gpu_memory_mb(use_pytorch_peak=False):
    if not torch.cuda.is_available():
        return 0.0 

    device_idx = torch.cuda.current_device()

    if use_pytorch_peak:
        try:
            peak_bytes = torch.cuda.max_memory_allocated(device_idx)
            return peak_bytes / (1024 * 1024)
        except Exception as e:
            print(f"Warning: Could not get PyTorch peak allocated memory: {e}", file=sys.stderr)
            return -1.0
    elif pynvml_available:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_bytes = mem_info.used
            pynvml.nvmlShutdown()
            return used_bytes / (1024 * 1024)
        except Exception as e:
            print(f"Warning: pynvml failed to get GPU memory: {e}. Falling back to PyTorch current.", file=sys.stderr)
            try:
                current_bytes = torch.cuda.memory_allocated(device_idx)
                return current_bytes / (1024 * 1024)
            except Exception as e_torch:
                print(f"Warning: Could not get PyTorch current allocated memory: {e_torch}", file=sys.stderr)
                return -1.0
    else:
        try:
            current_bytes = torch.cuda.memory_allocated(device_idx)
            print("Info: Reporting current PyTorch tensor memory (pynvml not available).", file=sys.stderr)
            return current_bytes / (1024 * 1024)
        except Exception as e:
            print(f"Warning: Could not get PyTorch current allocated memory: {e}", file=sys.stderr)
            return -1.0

def get_current_gpu_memory_mb():
    return get_gpu_memory_mb(use_pytorch_peak=True)

def get_peak_gpu_memory_mb():
     return get_gpu_memory_mb(use_pytorch_peak=False)

def calculate_accuracy_classification_cpu(predictions_tensor, targets_tensor):

    if predictions_tensor.numel() == 0 or targets_tensor.numel() == 0:
        return 0.0

    predictions_cpu = predictions_tensor.detach().cpu()
    targets_cpu = targets_tensor.detach().cpu()

    pred_indices = torch.argmax(predictions_cpu, dim=1)

    if targets_cpu.ndim == 2 and targets_cpu.shape[1] > 1:
        target_indices = torch.argmax(targets_cpu, dim=1)
    elif targets_cpu.ndim == 1:
        target_indices = targets_cpu
    else:
        print("Warning: Unexpected target tensor shape for accuracy calculation.", file=sys.stderr)
        return 0.0

    correct = (pred_indices == target_indices).sum().item()
    accuracy = correct / targets_cpu.size(0) if targets_cpu.size(0) > 0 else 0.0
    return accuracy

def calculate_mae_regression_cpu(predictions_tensor, targets_tensor):
    if predictions_tensor.numel() == 0 or targets_tensor.numel() == 0:
        return 0.0

    predictions_cpu = predictions_tensor.detach().cpu()
    targets_cpu = targets_tensor.detach().cpu()

    mae = torch.abs(predictions_cpu - targets_cpu).mean().item()
    return mae

def custom_loss_function(output, target):
    diff = output - target
    squared_error = torch.square(diff)

    weight = torch.tensor(1.0, device=output.device) + squared_error * torch.tensor(0.5, device=output.device)
    loss = weight * squared_error
    return loss.mean()

def calculate_custom_loss_cpu(output_list, target_list):
    total_loss = 0.0
    if not output_list or len(output_list) != len(target_list):
        return 0.0
    for i in range(len(output_list)):
        diff = output_list[i] - target_list[i]
        squared_error = diff * diff
        weight = 1.0 + squared_error * 0.5
        total_loss += weight * squared_error
    return total_loss / len(output_list) if output_list else 0.0

class CsvDataset(Dataset):
    def __init__(self, csv_path, feature_cols_indices, target_col_index, header_lines=1, normalize=True, delimiter=','):
        self.normalize = normalize
        self.features_host = []
        self.targets_host = [] 

        print(f"Manually loading CSV data from: {csv_path}")

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=delimiter)

                for _ in range(header_lines):
                    try:
                        next(reader)
                    except StopIteration:
                        raise RuntimeError(f"CSV file {csv_path} has fewer than {header_lines} header lines.")

                initial_rows = 0
                valid_rows = 0
                for row in reader:
                    initial_rows += 1
                    if not row: continue 
                    try:
                        feature_values = []
                        for idx in feature_cols_indices:
                            if idx >= len(row): raise IndexError("Feature column index out of range")
                            val_str = row[idx].strip().strip('"')
                            feature_values.append(float(val_str))
                        if target_col_index >= len(row): raise IndexError("Target column index out of range")
                        target_str = row[target_col_index].strip().strip('"')
                        target_value = float(target_str)

                        if any(math.isnan(v) for v in feature_values) or math.isnan(target_value):
                             print(f"Warning: Found NaN in row {initial_rows}. Skipping.", file=sys.stderr)
                             continue

                        self.features_host.append(feature_values)
                        self.targets_host.append([target_value])
                        valid_rows += 1

                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping invalid row {initial_rows} in {csv_path}. Error: {e}", file=sys.stderr)
                        continue

        except FileNotFoundError:
            raise RuntimeError(f"Failed to open dataset file: {csv_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading CSV {csv_path}: {e}")

        if valid_rows == 0:
             raise RuntimeError(f"Dataset loaded 0 valid samples after cleaning from: {csv_path}")

        print(f"Read {initial_rows} lines, loaded {valid_rows} valid samples.")

        self.features_np = np.array(self.features_host, dtype=np.float32)
        self.targets_np = np.array(self.targets_host, dtype=np.float32)

        self.feature_size_ = self.features_np.shape[1]
        self.target_size_ = self.targets_np.shape[1]

        if self.normalize:
            self.feature_means = np.mean(self.features_np, axis=0, dtype=np.float32)
            self.feature_stddevs = np.std(self.features_np, axis=0, dtype=np.float32)
            self.target_mean = np.mean(self.targets_np, axis=0, dtype=np.float32)
            self.target_stddev = np.std(self.targets_np, axis=0, dtype=np.float32)

            epsilon = 1e-8
            self.feature_stddevs[self.feature_stddevs < epsilon] = epsilon
            self.target_stddev[self.target_stddev < epsilon] = epsilon

            self.features_np = (self.features_np - self.feature_means) / self.feature_stddevs
            self.targets_np = (self.targets_np - self.target_mean) / self.target_stddev
            print(f"Normalized {valid_rows} samples.")
        else:
            self.feature_means = np.zeros(self.feature_size_, dtype=np.float32)
            self.feature_stddevs = np.ones(self.feature_size_, dtype=np.float32)
            self.target_mean = np.zeros(self.target_size_, dtype=np.float32)
            self.target_stddev = np.ones(self.target_size_, dtype=np.float32)
            print(f"Loaded {valid_rows} samples (not normalized).")

        print(f"  Features: {self.feature_size_}, Target: {self.target_size_}")


    def __len__(self):
        return len(self.features_np) 

    def __getitem__(self, index):
        features = torch.from_numpy(self.features_np[index])
        targets = torch.from_numpy(self.targets_np[index])
        return features, targets

    def get_num_features(self):
        return self.feature_size_

    def feature_size(self):
        return self.feature_size_

    def target_size(self):
        return self.target_size_

    def denormalize_targets(self, normalized_targets_tensor):
        if not self.normalize:
            return normalized_targets_tensor 

        if isinstance(normalized_targets_tensor, np.ndarray):
            normalized_targets_tensor = torch.from_numpy(normalized_targets_tensor)
        elif isinstance(normalized_targets_tensor, list):
             normalized_targets_tensor = torch.tensor(normalized_targets_tensor)

        target_mean_t = torch.from_numpy(self.target_mean).to(normalized_targets_tensor.device)
        target_stddev_t = torch.from_numpy(self.target_stddev).to(normalized_targets_tensor.device)
        return normalized_targets_tensor * target_stddev_t + target_mean_t


    def denormalize_features(self, normalized_features_tensor):
        if not self.normalize:
            return normalized_features_tensor
        if isinstance(normalized_features_tensor, np.ndarray):
            normalized_features_tensor = torch.from_numpy(normalized_features_tensor)
        elif isinstance(normalized_features_tensor, list):
             normalized_features_tensor = torch.tensor(normalized_features_tensor)

        feature_means_t = torch.from_numpy(self.feature_means).to(normalized_features_tensor.device)
        feature_stddevs_t = torch.from_numpy(self.feature_stddevs).to(normalized_features_tensor.device)
        return normalized_features_tensor * feature_stddevs_t + feature_means_t


class BostonDataset(Dataset):
    def __init__(self, features, targets, target_mean, target_std):
        if not isinstance(features, torch.Tensor):
            self.features = torch.tensor(features, dtype=torch.float32)
        else:
            self.features = features.float()

        if not isinstance(targets, torch.Tensor):
            self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
        else:
            self.targets = targets.float().view(-1, 1)

        if self.features.shape[0] != self.targets.shape[0]:
            raise ValueError(f"Number of samples mismatch: Features ({self.features.shape[0]}) vs Targets ({self.targets.shape[0]})")

        self.target_mean = torch.tensor(target_mean, dtype=torch.float32)
        self.target_std = torch.tensor(target_std, dtype=torch.float32)

        self.epsilon = 1e-8

        print(f"BostonDataset created with {len(self.features)} samples.")
        print(f"  Feature shape: {self.features.shape}")
        print(f"  Target shape: {self.targets.shape}")
        print(f"  Target Mean (for denorm): {self.target_mean.item():.4f}")
        print(f"  Target Std (for denorm): {self.target_std.item():.4f}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        return self.features[idx], self.targets[idx] 

    def get_num_features(self):
        return self.features.shape[1] if len(self.features.shape) > 1 else 0

    def denormalize_targets(self, normalized_targets_tensor):
        """Denormalizes target predictions using the stored mean and std."""
        if self.target_std is None or self.target_mean is None:
            print("Warning: Target normalization stats not available, returning original tensor.", file=sys.stderr)
            return normalized_targets_tensor

        std_tensor = self.target_std.to(normalized_targets_tensor.device)
        mean_tensor = self.target_mean.to(normalized_targets_tensor.device)

        denormalized = normalized_targets_tensor * (std_tensor + self.epsilon) + mean_tensor
        return denormalized
# California Dataset (Uses manual CsvDataset)
class CaliforniaDataset(Dataset):
    """
    Holds preprocessed California Housing data (features and targets as tensors)
    and target normalization statistics for denormalization.
    """
    def __init__(self, features, targets, target_mean, target_std):
        if not isinstance(features, torch.Tensor):
            self.features = torch.tensor(features, dtype=torch.float32)
        else:
            self.features = features.float() # Ensure correct type

        if not isinstance(targets, torch.Tensor):
            # Ensure targets are [N, 1] for consistency before potentially squeezing
            self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
        else:
            self.targets = targets.float().view(-1, 1) # Ensure correct type and shape

        if self.features.shape[0] != self.targets.shape[0]:
            raise ValueError(f"Number of samples mismatch: Features ({self.features.shape[0]}) vs Targets ({self.targets.shape[0]})")

        # Store normalization stats for denormalization of the TARGET variable
        # Ensure they are tensors for potential device transfer
        self.target_mean = torch.tensor(target_mean, dtype=torch.float32)
        self.target_std = torch.tensor(target_std, dtype=torch.float32)

        # Add a small epsilon to std dev to prevent division by zero during denormalization
        self.epsilon = 1e-8

        print(f"CaliforniaDataset created with {len(self.features)} samples.")
        print(f"  Feature shape: {self.features.shape}")
        print(f"  Target shape: {self.targets.shape}")
        print(f"  Target Mean (for denorm): {self.target_mean.item():.4f}")
        print(f"  Target Std (for denorm): {self.target_std.item():.4f}")


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Data is already preprocessed and tensorized
        return self.features[idx], self.targets[idx] # Return target as [1]

    def get_num_features(self):
        # Returns the number of features AFTER preprocessing (including one-hot)
        return self.features.shape[1] if len(self.features.shape) > 1 else 0

    def denormalize_targets(self, normalized_targets_tensor):
        """Denormalizes target predictions using the stored mean and std."""
        if self.target_std is None or self.target_mean is None:
            print("Warning: Target normalization stats not available, returning original tensor.", file=sys.stderr)
            return normalized_targets_tensor

        # Ensure stats are on the same device as the predictions
        std_tensor = self.target_std.to(normalized_targets_tensor.device)
        mean_tensor = self.target_mean.to(normalized_targets_tensor.device)

        # Perform denormalization: prediction = normalized_prediction * std + mean
        denormalized = normalized_targets_tensor * (std_tensor + self.epsilon) + mean_tensor
        return denormalized

# MNIST Dataset (Mimics MnistDataset.h)
# Uses torchvision.datasets.ImageFolder assuming the structure:
# base_path/MNIST - JPG - training/0/*.jpg, base_path/MNIST - JPG - training/1/*.jpg ...
# base_path/MNIST - JPG - testing/0/*.jpg, base_path/MNIST - JPG - testing/1/*.jpg ...
class MnistDatasetWrapper(Dataset):
    def __init__(self, base_dataset_path, mode='training', width=28, height=28):
        self.input_width = width
        self.input_height = height
        self.num_classes = 10
        self.image_channels = 1 # MNIST is grayscale
        self.feature_size_ = width * height * self.image_channels
        self.target_size_ = self.num_classes

        # --- Enforce MNIST dimensions ---
        # C++ likely assumes 28x28 grayscale
        if width != 28 or height != 28:
             print(f"Warning: Resizing MNIST images to {width}x{height}. Original is 28x28.", file=sys.stderr)
             # Keep the requested width/height for processing

        data_root = Path(base_dataset_path) / f"MNIST - JPG - {mode}"
        if not data_root.is_dir():
             raise RuntimeError(f"MNIST data directory not found: {data_root}")
        print(f"Manually loading MNIST JPG data from: {data_root}")

        self.images_host = [] # List to store image tensors (CHW format)
        self.labels_host = [] # List to store one-hot label tensors

        # --- Define transforms (applied manually after loading) ---
        # Normalize (MNIST specific)
        self.normalize_transform = transforms.Normalize((0.1307,), (0.3081,))

        # --- Find and load image files ---
        expected_classes = [str(i) for i in range(10)]
        found_classes = sorted([d.name for d in data_root.iterdir() if d.is_dir()])

        if set(found_classes) != set(expected_classes):
            print(f"Warning: Found class folders {found_classes} do not exactly match expected {expected_classes} in {data_root}", file=sys.stderr)

        for class_name in found_classes:
            class_dir = data_root / class_name
            try:
                label_index = int(class_name)
                if not (0 <= label_index < self.num_classes):
                    print(f"Warning: Skipping invalid class directory '{class_name}'", file=sys.stderr)
                    continue
            except ValueError:
                print(f"Warning: Skipping non-numeric class directory '{class_name}'", file=sys.stderr)
                continue

            one_hot_label = torch.zeros(self.num_classes, dtype=torch.float32)
            one_hot_label[label_index] = 1.0

            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) # Allow png too
            print(f"  Loading {len(image_files)} images for class {class_name}...")

            for img_path in image_files:
                try:
                    # Load image using OpenCV (mimics C++)
                    # IMREAD_GRAYSCALE ensures 1 channel
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise IOError(f"cv2.imread returned None for {img_path}")

                    # Resize if necessary
                    if img.shape[0] != self.input_height or img.shape[1] != self.input_width:
                        img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA) # Use INTER_AREA for shrinking

                    # Add channel dimension: (H, W) -> (H, W, C=1)
                    img = img[:, :, np.newaxis]

                    # Convert to float tensor [0, 1] and CHW format
                    # OpenCV loads HWC, need CHW: (H, W, 1) -> (1, H, W)
                    img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255.0)

                    # Apply normalization
                    img_tensor = self.normalize_transform(img_tensor)

                    # Flatten: (1, H, W) -> (H*W)
                    img_tensor = torch.flatten(img_tensor)

                    self.images_host.append(img_tensor)
                    self.labels_host.append(one_hot_label)

                except Exception as e:
                    print(f"Warning: Failed to load/process image {img_path}. Error: {e}. Skipping.", file=sys.stderr)

        if not self.images_host or not self.labels_host or len(self.images_host) != len(self.labels_host):
             raise RuntimeError(f"MNIST dataset loading failed or resulted in inconsistent data for {mode} set.")

        print(f"Finished manually loading MNIST {mode} set. Total images: {len(self.images_host)}")


    def __len__(self):
        return len(self.images_host)

    def __getitem__(self, index):
        if not (0 <= index < len(self.images_host)):
             raise IndexError(f"Index {index} out of range for dataset size {len(self.images_host)}")

        image_tensor = self.images_host[index] # Already flattened
        label_tensor = self.labels_host[index] # Already one-hot

        # Return label index for CrossEntropyLoss compatibility
        label_index = torch.argmax(label_tensor).item()
        return image_tensor, label_index
        # If strict one-hot needed: return image_tensor, label_tensor

    def feature_size(self):
        return self.feature_size_

    def target_size(self):
        return self.target_size_

    def get_num_classes(self):
        return self.num_classes


    def feature_size(self):
        return self.feature_size_

    def target_size(self):
        # If returning one-hot, it's num_classes. If returning index, it's conceptually 1.
        # Let's return num_classes to match C++ hint.
        return self.target_size_

    def get_num_classes(self):
        return self.num_classes

# CIFAR-10 Dataset (Mimics CifarDataset.h)
# Uses torchvision.datasets.CIFAR10
class CifarDatasetWrapper(Dataset):
    def __init__(self, base_dataset_path, mode='training', width=32, height=32):
        self.input_width = width
        self.input_height = height
        self.num_classes = 10
        self.image_channels = 3
        self.feature_size_ = width * height * self.image_channels
        self.target_size_ = self.num_classes
        is_train = (mode == 'training')

        # --- Enforce CIFAR-10 dimensions ---
        # The C++ version doesn't resize, it assumes 32x32.
        if width != 32 or height != 32:
            raise ValueError(f"CifarDatasetWrapper currently only supports 32x32 images like the C++ version. Requested: {width}x{height}")

        print(f"Manually loading CIFAR-10 data from: {base_dataset_path} ({mode})")

        self.images_host = [] # List to store image tensors (CHW format)
        self.labels_host = [] # List to store one-hot label tensors

        # --- Define normalization transform (applied after loading) ---
        # Matches the normalization used previously, but applied manually now
        self.normalize_transform = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                        std=[0.2023, 0.1994, 0.2010])

        # --- Load data from binary files ---
        data_path = Path(base_dataset_path)
        if not data_path.is_dir():
             raise RuntimeError(f"CIFAR-10 base directory not found: {base_dataset_path}")

        if is_train:
            for i in range(1, 6):
                batch_file = data_path / f"data_batch_{i}.bin"
                self._load_cifar_batch(batch_file)
        else: # Testing mode
            batch_file = data_path / "test_batch.bin"
            self._load_cifar_batch(batch_file)

        if not self.images_host or not self.labels_host or len(self.images_host) != len(self.labels_host):
             raise RuntimeError(f"CIFAR-10 dataset loading failed or resulted in inconsistent data for {mode} set.")

        print(f"Finished manually loading CIFAR-10 {mode} set. Total images: {len(self.images_host)}")

    def _load_cifar_batch(self, filename):
        """Loads a single CIFAR-10 batch file (binary format)."""
        # Constants from C++ version
        IMAGE_HEIGHT = 32
        IMAGE_WIDTH = 32
        IMAGE_CHANNELS = 3
        NUM_CLASSES = 10
        IMAGE_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS # 3072
        RECORD_BYTES = 1 + IMAGE_BYTES # 1 label byte + image bytes

        try:
            with open(filename, 'rb') as f:
                all_data = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"Failed to open CIFAR batch file: {filename}")
        except Exception as e:
            raise RuntimeError(f"Error reading CIFAR batch file {filename}: {e}")

        total_records = len(all_data) // RECORD_BYTES
        if len(all_data) % RECORD_BYTES != 0:
             print(f"Warning: File size {len(all_data)} is not a multiple of record size {RECORD_BYTES} for {filename}", file=sys.stderr)

        print(f"  Loading {total_records} images from {filename.name}...")

        for i in range(total_records):
            record_start = i * RECORD_BYTES
            record_data = all_data[record_start : record_start + RECORD_BYTES]

            if len(record_data) < RECORD_BYTES:
                print(f"Warning: Incomplete record {i} found in {filename.name}. Skipping.", file=sys.stderr)
                continue

            # --- Process Label ---
            label_byte = int(record_data[0])
            if not (0 <= label_byte < NUM_CLASSES):
                print(f"Warning: Invalid label byte {label_byte} in record {i} of {filename.name}. Skipping.", file=sys.stderr)
                continue

            one_hot_label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
            one_hot_label[label_byte] = 1.0
            self.labels_host.append(one_hot_label)

            # --- Process Image ---
            # Image bytes are stored as 3072 bytes: 1024 R, 1024 G, 1024 B
            image_bytes = record_data[1:]
            # Convert to numpy array and reshape: (Channels, Height, Width)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8).reshape(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)

            # Convert to float tensor [0, 1] (CHW format)
            # Note: PyTorch's ToTensor typically expects HWC, but since we read directly into CHW order...
            # We need to transpose if we want to use ToTensor, or just divide by 255.
            # Let's convert directly to float tensor and divide.
            image_tensor = torch.from_numpy(image_np).float().div(255.0)

            # Apply normalization
            image_tensor = self.normalize_transform(image_tensor)

            self.images_host.append(image_tensor)


    def __len__(self):
        return len(self.images_host)

    def __getitem__(self, index):
        # Mimics get_sample(index, out_features, out_targets)
        if not (0 <= index < len(self.images_host)):
             raise IndexError(f"Index {index} out of range for dataset size {len(self.images_host)}")

        image_tensor = self.images_host[index]
        label_tensor = self.labels_host[index] # Already one-hot

        # Return label index if needed for CrossEntropyLoss, otherwise one-hot
        # To strictly match C++, return one-hot. For PyTorch training, index is better.
        # Let's return index for practical use with nn.CrossEntropyLoss
        label_index = torch.argmax(label_tensor).item()
        return image_tensor, label_index
        # If strict one-hot needed: return image_tensor, label_tensor


    def feature_size(self):
        # Return flattened size like C++ version
        return self.feature_size_

    def target_size(self):
        # Return one-hot size like C++ version
        return self.target_size_

    def get_num_classes(self):
        return self.num_classes

# Butterfly Dataset (Mimics Butterfly_dataset.h)
class ButterflyDataset(Dataset):
    def __init__(self, csv_path, data_dir, width=224, height=224, rgb=True):
        self.dataset_dir = Path(data_dir)
        self.input_width = width
        self.input_height = height
        self.use_rgb = rgb
        self.image_channels = 3 if rgb else 1
        self.image_paths = []
        self.labels = [] # Store label strings
        self.label_to_index = {}
        self.index_to_label = {}
        self.num_classes = 0

        # --- Read metadata from CSV (using pandas as C++ likely reads a file too) ---
        csv_filepath = Path(csv_path)
        csv_filename = csv_filepath.name
        if csv_filename == "Training_set.csv":
            self.data_subfolder = "train"
        elif csv_filename == "Testing_set.csv":
            self.data_subfolder = "test"
        else:
            self.data_subfolder = ""
            print(f"Warning: Could not determine data subfolder from CSV name '{csv_filename}'. Looking in '{data_dir}'.", file=sys.stderr)

        print(f"Using data subfolder: '{self.data_subfolder}' relative to '{self.dataset_dir}'")

        try:
            # Keep pandas for reading the CSV metadata file itself
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise RuntimeError(f"Failed to open file: {csv_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading CSV {csv_path}: {e}")

        has_labels = 'label' in df.columns
        if has_labels:
            unique_labels_found = set(df['label'].astype(str).unique())
            self.num_classes = len(unique_labels_found)
            self.label_to_index = {label: idx for idx, label in enumerate(sorted(list(unique_labels_found)))}
            self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
            print(f"Found {self.num_classes} unique classes.")
        else:
            print("No 'label' column found in CSV, assuming test set or unlabeled data.")
            self.num_classes = 0

        # --- Store image paths and labels ---
        for index, row in df.iterrows():
            filename = row['filename']
            label_str = str(row['label']) if has_labels else ""
            relative_path = Path(self.data_subfolder) / filename
            full_path = self.dataset_dir / relative_path
            if full_path.is_file():
                self.image_paths.append(str(full_path))
                self.labels.append(label_str)
            else:
                print(f"Warning: File not found or not a regular file: {full_path}", file=sys.stderr)

        self.feature_size_ = self.input_width * self.input_height * self.image_channels
        self.target_size_ = self.num_classes if self.num_classes > 0 else 0

        # --- Define transforms (applied manually after loading) ---
        # Normalization (example: ImageNet stats if needed)
        # self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        print(f"Stored {len(self.image_paths)} image paths. Using {'RGB' if self.use_rgb else 'grayscale'} images.")
        print(f"Feature size hint: {self.feature_size_}, Target size hint: {self.target_size_}")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label_str = self.labels[index]

        try:
            # --- Load image using OpenCV (mimics C++) ---
            read_mode = cv2.IMREAD_COLOR if self.use_rgb else cv2.IMREAD_GRAYSCALE
            img = cv2.imread(img_path, read_mode)
            if img is None:
                raise IOError(f"cv2.imread returned None for {img_path}")

            # OpenCV loads BGR by default if color, convert to RGB
            if self.use_rgb and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Add channel dimension if grayscale: (H, W) -> (H, W, 1)
            if not self.use_rgb:
                img = img[:, :, np.newaxis]

            # --- Apply transforms manually ---
            # Resize
            if img.shape[0] != self.input_height or img.shape[1] != self.input_width:
                img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
                # Add channel dim back if resize removed it for grayscale
                if not self.use_rgb and len(img.shape) == 2:
                    img = img[:, :, np.newaxis]

            # Convert to float tensor [0, 1] and CHW format
            # OpenCV loads HWC, need CHW: (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255.0)

            # Apply normalization if defined
            # if hasattr(self, 'normalize_transform'):
            #     img_tensor = self.normalize_transform(img_tensor)

        except Exception as e:
            print(f"Warning: Failed to load or process image: {img_path}. Error: {e}. Returning blank.", file=sys.stderr)
            img_tensor = torch.zeros((self.image_channels, self.input_height, self.input_width))
            label_index = -1 # Indicate error

        else: # Only process label if image loading succeeded
            # Get label index
            if self.num_classes > 0 and label_str in self.label_to_index:
                label_index = self.label_to_index[label_str]
            elif self.num_classes > 0:
                 print(f"Warning: Label '{label_str}' not found in mapping for image {img_path}. Assigning index -1.", file=sys.stderr)
                 label_index = -1
            else:
                 label_index = -1 # No labels expected

        # Return image tensor and label index
        return img_tensor, label_index


    def feature_size(self):
        return self.feature_size_

    def target_size(self):
        return self.target_size_

    def get_num_classes(self):
        return self.num_classes

    def get_class_name(self, index):
        return self.index_to_label.get(index, "Unknown")

# Generic Image Dataset (Mimics ImageDataset.h used in conv.cpp)
# Likely just ImageFolder
class ImageDatasetWrapper(Dataset):
     def __init__(self, data_dir, width=32, height=32, use_grayscale=True, for_autoencoder=True):
        self.input_width = width
        self.input_height = height
        self.use_grayscale = use_grayscale
        self.image_channels = 1 if use_grayscale else 3
        self.for_autoencoder = for_autoencoder # Flag to return image as target
        self.root_dir = Path(data_dir)

        print(f"Manually loading generic image data from: {self.root_dir}")
        print(f"Image Size: {width}x{height}, Grayscale: {use_grayscale}, For AE: {for_autoencoder}")

        self.images_host = [] # List to store image tensors (CHW)
        self.labels_host = [] # List to store label indices (or copy of image tensor for AE)
        self.classes = []
        self.class_to_idx = {}

        # --- Find classes and images ---
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        subdirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        if not subdirs:
             # If no subdirectories, assume images are directly in root_dir
             print(f"No subdirectories found in {self.root_dir}. Loading images directly (no classes assigned).")
             self.num_classes = 0
             image_files = []
             for ext in image_extensions:
                 image_files.extend(self.root_dir.glob(f'*{ext}'))
             print(f"  Found {len(image_files)} images in root directory.")
             self._load_images_from_list(image_files, label_index=-1) # Assign dummy label index
        else:
             # Assume subdirectories are classes
             self.classes = [d.name for d in subdirs]
             self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
             self.num_classes = len(self.classes)
             print(f"Found {self.num_classes} classes: {self.classes}")

             for class_name in self.classes:
                 class_idx = self.class_to_idx[class_name]
                 class_dir = self.root_dir / class_name
                 image_files = []
                 for ext in image_extensions:
                     image_files.extend(class_dir.glob(f'*{ext}'))
                 print(f"  Loading {len(image_files)} images for class '{class_name}' (index {class_idx})...")
                 self._load_images_from_list(image_files, label_index=class_idx)


        if not self.images_host:
             raise RuntimeError(f"No images found or loaded from {self.root_dir}")

        self.feature_size_ = self.input_width * self.input_height * self.image_channels
        # Target size depends on usage
        self.target_size_ = self.feature_size_ if self.for_autoencoder else self.num_classes

        print(f"Finished loading {len(self.images_host)} images.")
        print(f"Feature size hint: {self.feature_size_}, Target size hint: {self.target_size_}")


     def _load_images_from_list(self, image_files, label_index):
         """Helper to load and process images from a list of paths."""
         for img_path in image_files:
             try:
                 # Load image using OpenCV
                 read_mode = cv2.IMREAD_GRAYSCALE if self.use_grayscale else cv2.IMREAD_COLOR
                 img = cv2.imread(str(img_path), read_mode)
                 if img is None:
                     raise IOError(f"cv2.imread returned None for {img_path}")

                 # Convert BGR to RGB if color
                 if not self.use_grayscale and img.shape[2] == 3:
                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                 # Add channel dim if grayscale
                 if self.use_grayscale:
                     img = img[:, :, np.newaxis]

                 # Resize
                 if img.shape[0] != self.input_height or img.shape[1] != self.input_width:
                     img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
                     if self.use_grayscale and len(img.shape) == 2: # Ensure channel dim exists
                         img = img[:, :, np.newaxis]

                 # Convert to float tensor [0, 1] and CHW format
                 img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255.0)

                 # Apply normalization if needed (define self.normalize_transform if required)
                 # if hasattr(self, 'normalize_transform'):
                 #     img_tensor = self.normalize_transform(img_tensor)

                 self.images_host.append(img_tensor)

                 # Store label index or image tensor for AE
                 if self.for_autoencoder:
                     self.labels_host.append(img_tensor.clone()) # Store a copy for AE target
                 else:
                     self.labels_host.append(label_index) # Store class index

             except Exception as e:
                 print(f"Warning: Failed to load/process image {img_path}. Error: {e}. Skipping.", file=sys.stderr)


     def __len__(self):
         return len(self.images_host)

     def __getitem__(self, index):
         if not (0 <= index < len(self.images_host)):
             raise IndexError(f"Index {index} out of range for dataset size {len(self.images_host)}")

         img_tensor = self.images_host[index]
         target = self.labels_host[index] # This is either label index or image tensor

         return img_tensor, target


     def feature_size(self):
         return self.feature_size_

     def target_size(self):
         # Return hint based on intended use
         return self.target_size_

     def get_num_classes(self):
         # Return 0 if no classes were found or if used for AE?
         # Let's return the number of folders found.
         return self.num_classes


# --- Model Implementations (nn.Module) ---

# MNIST Classifier (Mimics architecture in mnist_train.cpp)
class MnistClassifier(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        # C++ version: Linear(INPUT_SIZE, 128, "relu"), Linear(128, 64, "relu"), Linear(64, NUM_CLASSES, "softmax")
        self.layer1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, num_classes)
        # Softmax is often included in the loss function (CrossEntropyLoss)
        # If needed explicitly: self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Assumes input x is already flattened (N, input_size)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        # x = self.softmax(x) # Apply softmax if not using CrossEntropyLoss
        return x

# CIFAR-10 Classifier (Mimics architecture in cifar_train.cpp)
class CifarClassifier(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        # C++ version: Conv2d(3, 32, kernel=3, stride=1, padding=1, "relu"), Conv2d(32, 64, k=3, s=1, p=1, "relu"), MaxPool2d(k=2, s=2),
        #              Conv2d(64, 128, k=3, s=1, p=1, "relu"), Conv2d(128, 128, k=3, s=1, p=1, "relu"), MaxPool2d(k=2, s=2),
        #              Flatten(), Linear(128*8*8, 512, "relu"), Linear(512, NUM_CLASSES, "softmax")
        # Note: Input is 32x32
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 16x16 -> 16x16
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 16x16 -> 16x16
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8

        self.flatten = nn.Flatten()
        # Calculate flattened size: 128 channels * 8 height * 8 width
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        # Softmax often included in loss

    def forward(self, x):
        # Input x shape: (N, C, H, W) = (N, 3, 32, 32)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        return x

# Regression Model (Mimics architectures in boston_regression.cpp, california_regression.cpp)
class RegressionModel(nn.Module):
    def __init__(self, num_features, output_size=1):
        super().__init__()
        # C++ Boston: Linear(num_features, 64, "relu"), Linear(64, 32, "relu"), Linear(32, output_size, "none")
        # C++ California: Linear(num_features, 128, "relu"), Linear(128, 64, "relu"), Linear(64, 32, "relu"), Linear(32, output_size, "none")
        # Let's make it configurable or use the larger California one as default
        self.layer1 = nn.Linear(num_features, 128) # 64 for Boston
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64) # 32 for Boston
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 32) # Skip for Boston
        self.relu3 = nn.ReLU() # Skip for Boston
        self.layer4 = nn.Linear(32, output_size) # Use layer3 for Boston: nn.Linear(32, output_size)

        # Simpler Boston version:
        # self.layer1 = nn.Linear(num_features, 64)
        # self.relu1 = nn.ReLU()
        # self.layer2 = nn.Linear(64, 32)
        # self.relu2 = nn.ReLU()
        # self.layer3 = nn.Linear(32, output_size)


    def forward(self, x):
        # California version
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.layer4(x)

        # Boston version:
        # x = self.layer1(x)
        # x = self.relu1(x)
        # x = self.layer2(x)
        # x = self.relu2(x)
        # x = self.layer3(x)
        return x

# Convolutional Autoencoder (Mimics architecture in conv.cpp)
class ConvAutoencoder(nn.Module):
     def __init__(self, input_channels=1, input_height=32, input_width=32):
        super().__init__()
        # C++ version:
        # Encoder: Conv(in_channels, 16, k=3, s=1, p=1, "relu"), MaxPool(k=2, s=2), Conv(16, 8, k=3, s=1, p=1, "relu"), MaxPool(k=2, s=2)
        # Decoder: ConvTranspose(8, 16, k=2, s=2, p=0, "relu"), ConvTranspose(16, out_channels, k=2, s=2, p=0, "sigmoid")
        # Note: Output channels should match input channels
        output_channels = input_channels
        h, w = input_height, input_width

        # Encoder
        self.enc_conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1) # H -> H, W -> W
        self.enc_relu1 = nn.ReLU()
        self.enc_pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # H -> H/2, W -> W/2
        self.enc_conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1) # H/2 -> H/2, W/2 -> W/2
        self.enc_relu2 = nn.ReLU()
        self.enc_pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # H/2 -> H/4, W/2 -> W/4

        # Decoder
        # Input to decoder: (N, 8, H/4, W/4)
        self.dec_tconv1 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2, padding=0) # H/4 -> H/2, W/4 -> W/2
        self.dec_relu1 = nn.ReLU()
        self.dec_tconv2 = nn.ConvTranspose2d(16, output_channels, kernel_size=2, stride=2, padding=0) # H/2 -> H, W/2 -> W
        self.dec_sigmoid = nn.Sigmoid() # Output between 0 and 1

     def forward(self, x):
        # Encoder
        x = self.enc_conv1(x)
        x = self.enc_relu1(x)
        x = self.enc_pool1(x)
        x = self.enc_conv2(x)
        x = self.enc_relu2(x)
        x = self.enc_pool2(x)
        # Decoder
        x = self.dec_tconv1(x)
        x = self.dec_relu1(x)
        x = self.dec_tconv2(x)
        x = self.dec_sigmoid(x)
        return x


# --- Training Functions ---

# Mimics print_mnist_predictions / print_cifar_predictions
def print_classification_predictions(targets_host_indices, outputs_host, num_samples_to_show, batch_size, num_classes, dataset_name="Dataset", class_names=None):
    if outputs_host is None or targets_host_indices is None:
        print("Warning: Cannot print predictions due to missing host data.", file=sys.stderr)
        return

    num_samples_to_show = min(num_samples_to_show, batch_size)
    print(f"\n--- Sample {dataset_name} Predictions ---")
    header = "Sample | True Label | Predicted Label"
    if class_names:
        header += " | True Name | Predicted Name"
    # header += " | Output Scores (Optional)" # Keep it simpler
    print(header)
    print("-" * (len(header) + 20))

    outputs_host_np = outputs_host.numpy()
    targets_host_indices_np = targets_host_indices.numpy()

    pred_indices = np.argmax(outputs_host_np, axis=1)

    for i in range(num_samples_to_show):
        true_idx = int(targets_host_indices_np[i])
        pred_idx = int(pred_indices[i])

        line = f"{i:<6} | {true_idx:<10} | {pred_idx:<15}"
        if class_names:
            true_name = class_names[true_idx] if 0 <= true_idx < len(class_names) else "N/A"
            pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else "N/A"
            line += f" | {true_name:<9} | {pred_name:<14}"

        # Optionally print scores
        # scores = outputs_host_np[i]
        # scores_str = ", ".join([f"{s:.2f}" for s in scores])
        # line += f" | {scores_str}"
        print(line)

    print("-" * (len(header) + 20))


# Mimics mnist_train.cpp / cpu_mnist_train.cpp
def train_mnist_classifier(batch_size, num_epochs, dataset_path, image_height, image_width, use_grayscale, learning_rate=0.001, use_cuda_if_available=True):

    # --- Setup ---
    current_device = device if use_cuda_if_available and torch.cuda.is_available() else torch.device("cpu")
    is_cuda = current_device.type == 'cuda'
    mode_tag = "(GPU)" if is_cuda else "(CPU)"

    print(f"\n=== MNIST Classifier Training {mode_tag} ===")
    print(f"Using CUDA: {'Yes' if is_cuda else 'No'}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    print(f"Dataset Path: {dataset_path}")

    # Force grayscale and standard size for MNIST
    use_grayscale = True
    image_height, image_width = 28, 28
    input_size = image_height * image_width

    program_start_time = time.perf_counter()
    epoch_times_s = []
    total_train_batches_processed = 0
    # --- Timing Accumulators (Total) ---
    total_data_load_s = 0.0
    total_to_device_s = 0.0
    total_forward_s = 0.0
    total_loss_compute_s = 0.0
    total_backward_s = 0.0
    total_update_s = 0.0
    total_acc_calc_s = 0.0

    # --- Data ---
    try:
        print("Loading datasets...")
        load_start_time = time.perf_counter()
        train_dataset = MnistDatasetWrapper(dataset_path, mode='training', width=image_width, height=image_height)
        test_dataset = MnistDatasetWrapper(dataset_path, mode='testing', width=image_width, height=image_height)
        load_end_time = time.perf_counter()
        print(f"  Dataset loading time: {load_end_time - load_start_time:.2f} s")
        if len(train_dataset) == 0 or len(test_dataset) == 0:
             raise RuntimeError("Loaded 0 samples for train or test set.")

    except Exception as e:
        print(f"Error loading MNIST dataset: {e}", file=sys.stderr)
        return

    num_workers = 4 if is_cuda else 0
    pin_memory = is_cuda
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    total_train_batches_per_epoch = len(train_loader)
    total_test_batches = len(test_loader)

    # --- Model, Optimizer, Loss ---
    try:
        network = MnistClassifier(input_size=input_size, num_classes=train_dataset.get_num_classes()).to(current_device)
        if not list(network.parameters()):
             raise RuntimeError("Network has no parameters.")
    except Exception as e:
        print(f"Error creating network: {e}", file=sys.stderr)
        return

    optimizer = optim.Adam(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    loss_type = "CrossEntropyLoss"
    print(f"Using Optimizer: Adam (LR={learning_rate})")
    print(f"Using Loss: {loss_type}")

    # --- Training Loop ---
    print("Starting training...")
    # print(f"{'Epoch':<6} | {'Batch':<10} | {'Loss':<12} | {'Accuracy':<10} | {'Time/Epoch (s)':<15} | {'GPU Mem (MB)':<12}")
    # print("-" * 70)

    train_start_overall = time.perf_counter()
    if is_cuda: torch.cuda.reset_peak_memory_stats(current_device)

    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        network.train()
        running_loss = 0.0
        running_accuracy = 0.0
        batches_processed_this_epoch = 0
        # --- Timing Accumulators (Epoch) ---
        epoch_data_load_s = 0.0
        epoch_to_device_s = 0.0
        epoch_forward_s = 0.0
        epoch_loss_compute_s = 0.0
        epoch_backward_s = 0.0
        epoch_update_s = 0.0
        epoch_acc_calc_s = 0.0

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        batch_iter_start_time = time.perf_counter() # Time batch iteration start

        for i, (inputs, targets) in enumerate(train_loader):
            batch_iter_end_time = time.perf_counter()
            epoch_data_load_s += (batch_iter_end_time - batch_iter_start_time)

            # --- Time To Device ---
            to_device_start = time.perf_counter()
            inputs, targets = inputs.to(current_device), targets.to(current_device)
            to_device_end = time.perf_counter()
            epoch_to_device_s += (to_device_end - to_device_start)
            # --- End Time To Device ---

            # Zero gradients
            optimizer.zero_grad()

            # --- Time Forward Pass ---
            fwd_start = time.perf_counter()
            outputs = network(inputs)
            fwd_end = time.perf_counter()
            epoch_forward_s += (fwd_end - fwd_start)
            # --- End Time Forward Pass ---

            # --- Time Loss Computation ---
            loss_start = time.perf_counter()
            loss = criterion(outputs, targets)
            loss_end = time.perf_counter()
            epoch_loss_compute_s += (loss_end - loss_start)
            # --- End Time Loss Computation ---

            # --- Time Backward Pass ---
            bwd_start = time.perf_counter()
            loss.backward()
            bwd_end = time.perf_counter()
            epoch_backward_s += (bwd_end - bwd_start)
            # --- End Time Backward Pass ---

            # --- Time Optimizer Step ---
            update_start = time.perf_counter()
            optimizer.step()
            update_end = time.perf_counter()
            epoch_update_s += (update_end - update_start)
            # --- End Time Optimizer Step ---

            # --- Metrics (and Time Accuracy Calc) ---
            batch_loss = loss.item()
            running_loss += batch_loss

            acc_calc_start = time.perf_counter()
            batch_accuracy = calculate_accuracy_classification_cpu(outputs, targets)
            acc_calc_end = time.perf_counter()
            epoch_acc_calc_s += (acc_calc_end - acc_calc_start)
            running_accuracy += batch_accuracy
            # --- End Time Accuracy Calc ---

            batches_processed_this_epoch += 1
            total_train_batches_processed += 1

            # Print progress
            if (i + 1) % 50 == 0 or (i + 1) == total_train_batches_per_epoch:
                avg_loss = running_loss / batches_processed_this_epoch
                avg_acc = running_accuracy / batches_processed_this_epoch
                print(f"  Batch {i+1:>4}/{total_train_batches_per_epoch:<5} Loss: {avg_loss:<12.4f} Acc: {avg_acc:<10.4f}", end='\r')

            batch_iter_start_time = time.perf_counter() # Reset for next iteration's data load time

        # End of Epoch
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times_s.append(epoch_duration)
        avg_epoch_loss = running_loss / batches_processed_this_epoch if batches_processed_this_epoch > 0 else 0
        avg_epoch_acc = running_accuracy / batches_processed_this_epoch if batches_processed_this_epoch > 0 else 0
        gpu_mem = get_current_gpu_memory_mb() if is_cuda else 0

        print() # Newline after batch progress
        print(f"  Train Loss: {avg_epoch_loss:<12.4f} | Train Acc: {avg_epoch_acc:<10.4f}")
        print(f"  Epoch Time: {epoch_duration:.2f} s")
        if batches_processed_this_epoch > 0:
             print(f"  Avg Batch Timings (ms): "
                   f"Load: {epoch_data_load_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"ToDev: {epoch_to_device_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Fwd: {epoch_forward_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Loss: {epoch_loss_compute_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Bwd: {epoch_backward_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Update: {epoch_update_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"AccCalc: {epoch_acc_calc_s * 1000 / batches_processed_this_epoch:.3f}")
        if is_cuda: print(f"  GPU Mem Used (MB): {gpu_mem:.2f}")

        # Accumulate totals
        total_data_load_s += epoch_data_load_s
        total_to_device_s += epoch_to_device_s
        total_forward_s += epoch_forward_s
        total_loss_compute_s += epoch_loss_compute_s
        total_backward_s += epoch_backward_s
        total_update_s += epoch_update_s
        total_acc_calc_s += epoch_acc_calc_s


    train_end_overall = time.perf_counter()
    total_train_time_s = train_end_overall - train_start_overall
    avg_epoch_time = sum(epoch_times_s) / len(epoch_times_s) if epoch_times_s else 0
    print("-" * 70)
    print(f"Training complete! Total Training Time: {total_train_time_s:.2f} s, Avg Epoch Time: {avg_epoch_time:.2f} s")

    # --- Testing Phase ---
    print(f"\n--- Testing {mode_tag} ---")
    network.eval() # Set model to evaluation mode
    test_loss = 0.0
    test_accuracy = 0.0
    test_batches = 0
    all_test_targets_host = []
    all_test_outputs_host = []
    first_batch_targets = None
    first_batch_outputs = None

    test_start_time = time.perf_counter()

    with torch.no_grad(): # Disable gradient calculations
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(current_device), targets.to(current_device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            test_accuracy += calculate_accuracy_classification_cpu(outputs, targets) # CPU calc
            test_batches += 1

            # Store results from first batch for printing predictions (on CPU)
            if first_batch_targets is None:
                first_batch_targets = targets.cpu()
                first_batch_outputs = outputs.cpu()

            # Optionally store all if needed, but can consume memory
            # all_test_targets_host.append(targets.cpu())
            # all_test_outputs_host.append(outputs.cpu())

    test_end_time = time.perf_counter()
    test_duration_s = test_end_time - test_start_time

    avg_test_loss = test_loss / test_batches if test_batches > 0 else 0
    avg_test_acc = test_accuracy / test_batches if test_batches > 0 else 0

    print(f"Test Results:")
    print(f"  Average Loss: {avg_test_loss:.4f}")
    print(f"  Average Accuracy: {avg_test_acc:.4f}")
    print(f"  Test Duration: {test_duration_s:.2f} s")

    # Print sample predictions from the first test batch
    if first_batch_targets is not None and first_batch_outputs is not None:
         print_classification_predictions(
             targets_host_indices=first_batch_targets,
             outputs_host=first_batch_outputs,
             num_samples_to_show=10,
             batch_size=first_batch_targets.size(0),
             num_classes=train_dataset.get_num_classes(),
             dataset_name="MNIST Test"
         )

    # --- Final Stats ---
    peak_cpu_mem = get_peak_cpu_memory_mb()
    peak_gpu_mem = get_peak_gpu_memory_mb() if is_cuda else 0
    avg_latency_ms = (total_train_time_s * 1000 / total_train_batches_processed) if total_train_batches_processed > 0 else 0
    throughput_samples_s = (total_train_batches_processed * batch_size / total_train_time_s) if total_train_time_s > 0 else 0

    print("\n--- Final Summary ---")
    print(f"Test Loss:               {avg_test_loss:.4f}")
    print(f"Test Accuracy:           {avg_test_acc:.4f}")
    print("--- Performance Metrics ---")
    print(f"Total Train Time (s):    {total_train_time_s:.2f}")
    print(f"Avg. Latency/Batch (ms): {avg_latency_ms:.2f}")
    print(f"Throughput (Samples/sec):{throughput_samples_s:.0f}")
    print(f"Avg. Time per Epoch (s): {avg_epoch_time:.2f}")
    if peak_cpu_mem >= 0: print(f"Peak CPU Memory (MB):    {peak_cpu_mem:.2f} (approx)")
    else: print("Peak CPU Memory (MB):    N/A")
    if is_cuda: print(f"Peak GPU Memory (MB):    {peak_gpu_mem:.2f}")

    if total_train_batches_processed > 0:
        print("--- Avg Batch Breakdown (ms) ---")
        print(f"  Data Load: {total_data_load_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  To Device: {total_to_device_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Forward:   {total_forward_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Loss Comp: {total_loss_compute_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Backward:  {total_backward_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Update:    {total_update_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Acc Calc:  {total_acc_calc_s * 1000 / total_train_batches_processed:.3f}")
    print("-" * 25)

    # Clean up GPU memory
    del network, train_loader, test_loader, inputs, targets, outputs, loss
    if is_cuda: torch.cuda.empty_cache()
    gc.collect()


# Mimics cifar_train.cpp / cpu_cifar_train.cpp
def train_cifar_classifier(batch_size, num_epochs, dataset_path, image_height, image_width, use_grayscale, learning_rate=0.001, use_cuda_if_available=True):

    # --- Setup ---
    current_device = device if use_cuda_if_available and torch.cuda.is_available() else torch.device("cpu")
    is_cuda = current_device.type == 'cuda'
    mode_tag = "(GPU)" if is_cuda else "(CPU)"

    print(f"\n=== CIFAR-10 Classifier Training {mode_tag} ===")
    print(f"Using CUDA: {'Yes' if is_cuda else 'No'}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    print(f"Dataset Path: {dataset_path}")

    # Force color and standard size for CIFAR-10
    use_grayscale = False
    image_height, image_width = 32, 32
    input_channels = 3

    program_start_time = time.perf_counter()
    epoch_times_s = []
    total_train_batches_processed = 0
    # --- Timing Accumulators (Total) ---
    total_data_load_s = 0.0
    total_to_device_s = 0.0
    total_forward_s = 0.0
    total_loss_compute_s = 0.0
    total_backward_s = 0.0
    total_update_s = 0.0
    total_acc_calc_s = 0.0

    # --- Data ---
    try:
        print("Loading datasets...")
        load_start_time = time.perf_counter()
        train_dataset = CifarDatasetWrapper(dataset_path, mode='training', width=image_width, height=image_height)
        test_dataset = CifarDatasetWrapper(dataset_path, mode='testing', width=image_width, height=image_height)
        load_end_time = time.perf_counter()
        print(f"  Dataset loading time: {load_end_time - load_start_time:.2f} s")
        if len(train_dataset) == 0 or len(test_dataset) == 0:
             raise RuntimeError("Loaded 0 samples for train or test set.")

    except Exception as e:
        print(f"Error loading CIFAR-10 dataset: {e}", file=sys.stderr)
        return

    num_workers = 4 if is_cuda else 0
    pin_memory = is_cuda
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    total_train_batches_per_epoch = len(train_loader)
    total_test_batches = len(test_loader)

    # --- Model, Optimizer, Loss ---
    try:
        network = CifarClassifier(input_channels=input_channels, num_classes=train_dataset.get_num_classes()).to(current_device)
        if not list(network.parameters()):
             raise RuntimeError("Network has no parameters.")
    except Exception as e:
        print(f"Error creating network: {e}", file=sys.stderr)
        return

    optimizer = optim.Adam(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    loss_type = "CrossEntropyLoss"
    print(f"Using Optimizer: Adam (LR={learning_rate})")
    print(f"Using Loss: {loss_type}")

    # --- Training Loop ---
    print("Starting training...")
    # print(f"{'Epoch':<6} | {'Batch':<10} | {'Loss':<12} | {'Accuracy':<10} | {'Time/Epoch (s)':<15} | {'GPU Mem (MB)':<12}")
    # print("-" * 70)

    train_start_overall = time.perf_counter()
    if is_cuda: torch.cuda.reset_peak_memory_stats(current_device)

    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        network.train()
        running_loss = 0.0
        running_accuracy = 0.0
        batches_processed_this_epoch = 0
        # --- Timing Accumulators (Epoch) ---
        epoch_data_load_s = 0.0
        epoch_to_device_s = 0.0
        epoch_forward_s = 0.0
        epoch_loss_compute_s = 0.0
        epoch_backward_s = 0.0
        epoch_update_s = 0.0
        epoch_acc_calc_s = 0.0

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        batch_iter_start_time = time.perf_counter()

        for i, (inputs, targets) in enumerate(train_loader):
            batch_iter_end_time = time.perf_counter()
            epoch_data_load_s += (batch_iter_end_time - batch_iter_start_time)

            # --- Time To Device ---
            to_device_start = time.perf_counter()
            inputs, targets = inputs.to(current_device), targets.to(current_device)
            to_device_end = time.perf_counter()
            epoch_to_device_s += (to_device_end - to_device_start)
            # --- End Time To Device ---

            optimizer.zero_grad()

            # --- Time Forward Pass ---
            fwd_start = time.perf_counter()
            outputs = network(inputs)
            fwd_end = time.perf_counter()
            epoch_forward_s += (fwd_end - fwd_start)
            # --- End Time Forward Pass ---

            # --- Time Loss Computation ---
            loss_start = time.perf_counter()
            loss = criterion(outputs, targets)
            loss_end = time.perf_counter()
            epoch_loss_compute_s += (loss_end - loss_start)
            # --- End Time Loss Computation ---

            # --- Time Backward Pass ---
            bwd_start = time.perf_counter()
            loss.backward()
            bwd_end = time.perf_counter()
            epoch_backward_s += (bwd_end - bwd_start)
            # --- End Time Backward Pass ---

            # --- Time Optimizer Step ---
            update_start = time.perf_counter()
            optimizer.step()
            update_end = time.perf_counter()
            epoch_update_s += (update_end - update_start)
            # --- End Time Optimizer Step ---

            # --- Metrics (and Time Accuracy Calc) ---
            batch_loss = loss.item()
            running_loss += batch_loss

            acc_calc_start = time.perf_counter()
            batch_accuracy = calculate_accuracy_classification_cpu(outputs, targets)
            acc_calc_end = time.perf_counter()
            epoch_acc_calc_s += (acc_calc_end - acc_calc_start)
            running_accuracy += batch_accuracy
            # --- End Time Accuracy Calc ---

            batches_processed_this_epoch += 1
            total_train_batches_processed += 1

            if (i + 1) % 50 == 0 or (i + 1) == total_train_batches_per_epoch:
                avg_loss = running_loss / batches_processed_this_epoch
                avg_acc = running_accuracy / batches_processed_this_epoch
                print(f"  Batch {i+1:>4}/{total_train_batches_per_epoch:<5} Loss: {avg_loss:<12.4f} Acc: {avg_acc:<10.4f}", end='\r')

            batch_iter_start_time = time.perf_counter() # Reset for next iteration

        # End of Epoch
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times_s.append(epoch_duration)
        avg_epoch_loss = running_loss / batches_processed_this_epoch if batches_processed_this_epoch > 0 else 0
        avg_epoch_acc = running_accuracy / batches_processed_this_epoch if batches_processed_this_epoch > 0 else 0
        gpu_mem = get_current_gpu_memory_mb() if is_cuda else 0

        print() # Newline after batch progress
        print(f"  Train Loss: {avg_epoch_loss:<12.4f} | Train Acc: {avg_epoch_acc:<10.4f}")
        print(f"  Epoch Time: {epoch_duration:.2f} s")
        if batches_processed_this_epoch > 0:
             print(f"  Avg Batch Timings (ms): "
                   f"Load: {epoch_data_load_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"ToDev: {epoch_to_device_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Fwd: {epoch_forward_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Loss: {epoch_loss_compute_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Bwd: {epoch_backward_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Update: {epoch_update_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"AccCalc: {epoch_acc_calc_s * 1000 / batches_processed_this_epoch:.3f}")
        if is_cuda: print(f"  GPU Mem Used (MB): {gpu_mem:.2f}")

        # Accumulate totals
        total_data_load_s += epoch_data_load_s
        total_to_device_s += epoch_to_device_s
        total_forward_s += epoch_forward_s
        total_loss_compute_s += epoch_loss_compute_s
        total_backward_s += epoch_backward_s
        total_update_s += epoch_update_s
        total_acc_calc_s += epoch_acc_calc_s

    train_end_overall = time.perf_counter()
    total_train_time_s = train_end_overall - train_start_overall
    avg_epoch_time = sum(epoch_times_s) / len(epoch_times_s) if epoch_times_s else 0
    print("-" * 70)
    print(f"Training complete! Total Training Time: {total_train_time_s:.2f} s, Avg Epoch Time: {avg_epoch_time:.2f} s")

    # --- Testing Phase ---
    print(f"\n--- Testing {mode_tag} ---")
    network.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    test_batches = 0
    first_batch_targets = None
    first_batch_outputs = None
    cifar_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
    
    test_start_time = time.perf_counter()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(current_device), targets.to(current_device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            test_accuracy += calculate_accuracy_classification_cpu(outputs, targets) # CPU calc
            test_batches += 1

            if first_batch_targets is None:
                first_batch_targets = targets.cpu()
                first_batch_outputs = outputs.cpu()

    test_end_time = time.perf_counter()
    test_duration_s = test_end_time - test_start_time

    avg_test_loss = test_loss / test_batches if test_batches > 0 else 0
    avg_test_acc = test_accuracy / test_batches if test_batches > 0 else 0

    print(f"Test Results:")
    print(f"  Average Loss: {avg_test_loss:.4f}")
    print(f"  Average Accuracy: {avg_test_acc:.4f}")
    print(f"  Test Duration: {test_duration_s:.2f} s")

    if first_batch_targets is not None and first_batch_outputs is not None:
         print_classification_predictions(
             targets_host_indices=first_batch_targets,
             outputs_host=first_batch_outputs,
             num_samples_to_show=10,
             batch_size=first_batch_targets.size(0),
             num_classes=train_dataset.get_num_classes(),
             dataset_name="CIFAR-10 Test",
             class_names=cifar_class_names
         )

    # --- Final Stats ---
    peak_cpu_mem = get_peak_cpu_memory_mb()
    peak_gpu_mem = get_peak_gpu_memory_mb() if is_cuda else 0
    avg_latency_ms = (total_train_time_s * 1000 / total_train_batches_processed) if total_train_batches_processed > 0 else 0
    throughput_samples_s = (total_train_batches_processed * batch_size / total_train_time_s) if total_train_time_s > 0 else 0

    print("\n--- Final Summary ---")
    print(f"Test Loss:               {avg_test_loss:.4f}")
    print(f"Test Accuracy:           {avg_test_acc:.4f}")
    print("--- Performance Metrics ---")
    print(f"Total Train Time (s):    {total_train_time_s:.2f}")
    print(f"Avg. Latency/Batch (ms): {avg_latency_ms:.2f}")
    print(f"Throughput (Samples/sec):{throughput_samples_s:.0f}")
    print(f"Avg. Time per Epoch (s): {avg_epoch_time:.2f}")
    if peak_cpu_mem >= 0: print(f"Peak CPU Memory (MB):    {peak_cpu_mem:.2f} (approx)")
    else: print("Peak CPU Memory (MB):    N/A")
    if is_cuda: print(f"Peak GPU Memory (MB):    {peak_gpu_mem:.2f}")

    if total_train_batches_processed > 0:
        print("--- Avg Batch Breakdown (ms) ---")
        print(f"  Data Load: {total_data_load_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  To Device: {total_to_device_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Forward:   {total_forward_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Loss Comp: {total_loss_compute_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Backward:  {total_backward_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Update:    {total_update_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Acc Calc:  {total_acc_calc_s * 1000 / total_train_batches_processed:.3f}")
    print("-" * 25)
    del network, train_loader, test_loader, inputs, targets, outputs, loss
    if is_cuda: torch.cuda.empty_cache()
    gc.collect()


# Mimics boston_regression.cpp / cpu_boston_regression.cpp
def train_boston_regressor(batch_size, num_epochs, dataset_path, learning_rate, loss_type_str, use_cuda_if_available=True):

    # --- Setup ---
    current_device = device if use_cuda_if_available and torch.cuda.is_available() else torch.device("cpu")
    is_cuda = current_device.type == 'cuda'
    mode_tag = "(GPU)" if is_cuda else "(CPU)"
    test_split_ratio = 0.2 # Use 20% for testing
    random_seed = 42 # For reproducible splits

    print(f"\n=== Boston Housing Regression Training {mode_tag} ===")
    print(f"Using CUDA: {'Yes' if is_cuda else 'No'}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Loss: {loss_type_str}")
    print(f"Dataset Path: {dataset_path}")

    program_start_time = time.perf_counter()
    epoch_times_s = []
    total_train_batches_processed = 0
    # --- Timing Accumulators (Total) ---
    total_data_load_s = 0.0 # Includes preprocessing
    total_to_device_s = 0.0
    total_forward_s = 0.0
    total_loss_compute_s = 0.0
    total_backward_s = 0.0
    total_update_s = 0.0
    total_metric_calc_s = 0.0 # Placeholder

    # --- Data Loading and Preprocessing ---
    data_load_start = time.perf_counter()
    try:
        print("Loading dataset...")
        try:
            # Boston.csv has a header row and an index column.
            df = pd.read_csv(dataset_path, index_col=0)
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV '{dataset_path}': {e}")

        # Verify target column exists (case-insensitive check)
        target_col = None
        if 'medv' in df.columns:
            target_col = 'medv'
        elif 'MEDV' in df.columns:
            target_col = 'MEDV'
        else:
            raise RuntimeError(f"Target column 'medv' or 'MEDV' not found in DataFrame columns: {df.columns.tolist()}")
        print(f"Using target column: '{target_col}'")

        features_df = df.drop(target_col, axis=1)
        targets_series = df[target_col]

        # Boston dataset is typically all numerical
        numerical_cols = features_df.columns.tolist()
        print(f"Numerical features: {numerical_cols}")

        # Split data BEFORE imputation/scaling
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets_series, test_size=test_split_ratio, random_state=random_seed
        )
        print(f"Data split: {len(X_train)} train samples, {len(X_test)} test samples.")

        # Create preprocessing pipeline for numerical features
        # Use SimpleImputer just in case, though Boston usually has no NaNs
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')), # Mean imputation is common for Boston
            ('scaler', StandardScaler())
        ])

        # Apply preprocessing: fit on train, transform train and test
        print("Applying preprocessing (imputation, scaling)...")
        X_train_processed = numerical_pipeline.fit_transform(X_train)
        X_test_processed = numerical_pipeline.transform(X_test)
        print(f"Processed feature shapes: Train {X_train_processed.shape}, Test {X_test_processed.shape}")

        num_features = X_train_processed.shape[1]
        output_size = 1 # Regression target

        # Scale the target variable (y) separately
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

        # Extract target mean and std dev for denormalization later
        target_mean_ = target_scaler.mean_[0]
        target_std_ = target_scaler.scale_[0]

        # Create Dataset instances with PREPROCESSED data
        train_dataset = BostonDataset(X_train_processed, y_train_scaled, target_mean_, target_std_)
        test_dataset = BostonDataset(X_test_processed, y_test_scaled, target_mean_, target_std_)

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error during data loading or preprocessing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    data_load_end = time.perf_counter()
    total_data_load_s += (data_load_end - data_load_start)
    print(f"Data loading and preprocessing took: {total_data_load_s:.2f} s")

    num_workers = 4 if is_cuda else 0
    pin_memory = is_cuda
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    total_train_batches_per_epoch = len(train_loader)
    total_test_batches = len(test_loader)
    print(f"Train batches/epoch: {total_train_batches_per_epoch}, Test batches: {total_test_batches}")

    # --- Model, Optimizer, Loss ---
    try:
        network = RegressionModel(num_features=num_features, output_size=output_size).to(current_device)
        # Overwrite with Boston structure:
        network.layer1 = nn.Linear(num_features, 64).to(current_device)
        network.layer2 = nn.Linear(64, 32).to(current_device)
        network.layer3 = nn.Linear(32, output_size).to(current_device)
        if hasattr(network, 'layer4'): del network.layer4
        if hasattr(network, 'relu3'): del network.relu3
        def boston_forward(self, x):
             x = self.layer1(x); x = self.relu1(x)
             x = self.layer2(x); x = self.relu2(x)
             x = self.layer3(x)
             return x
        network.forward = boston_forward.__get__(network, RegressionModel)

        if not list(network.parameters()):
             raise RuntimeError("Network has no parameters.")
    except Exception as e:
        print(f"Error creating network: {e}", file=sys.stderr)
        return

    optimizer = optim.Adam(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    if loss_type_str.lower() == "mse": criterion = nn.MSELoss()
    elif loss_type_str.lower() == "mae": criterion = nn.L1Loss()
    elif loss_type_str.lower() == "custom": criterion = custom_loss_function
    else:
        print(f"Warning: Unsupported loss type '{loss_type_str}'. Defaulting to MSE.", file=sys.stderr)
        criterion = nn.MSELoss(); loss_type_str = "MSE"

    print(f"Using Optimizer: Adam (LR={learning_rate})")
    print(f"Using Loss: {loss_type_str}")

    # --- Training Loop ---
    print("Starting training...")
    # print(f"{'Epoch':<6} | {'Batch':<10} | {'Loss':<12} | {'Time/Epoch (s)':<15} | {'GPU Mem (MB)':<12}")
    # print("-" * 60)

    train_start_overall = time.perf_counter()
    if is_cuda: torch.cuda.reset_peak_memory_stats(current_device)

    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        network.train()
        running_loss = 0.0
        batches_processed_this_epoch = 0
        # --- Timing Accumulators (Epoch) ---
        epoch_data_load_s = 0.0
        epoch_to_device_s = 0.0
        epoch_forward_s = 0.0
        epoch_loss_compute_s = 0.0
        epoch_backward_s = 0.0
        epoch_update_s = 0.0
        epoch_metric_calc_s = 0.0 # Placeholder

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        batch_iter_start_time = time.perf_counter()

        for i, (inputs, targets) in enumerate(train_loader):
            batch_iter_end_time = time.perf_counter()
            epoch_data_load_s += (batch_iter_end_time - batch_iter_start_time)

            # --- Time To Device ---
            to_device_start = time.perf_counter()
            inputs, targets = inputs.to(current_device), targets.to(current_device)
            to_device_end = time.perf_counter()
            epoch_to_device_s += (to_device_end - to_device_start)
            # --- End Time To Device ---

            optimizer.zero_grad()

            # --- Time Forward Pass ---
            fwd_start = time.perf_counter()
            outputs = network(inputs)
            fwd_end = time.perf_counter()
            epoch_forward_s += (fwd_end - fwd_start)
            # --- End Time Forward Pass ---

            # --- Time Loss Computation ---
            loss_start = time.perf_counter()
            loss = criterion(outputs, targets)
            loss_end = time.perf_counter()
            epoch_loss_compute_s += (loss_end - loss_start)
            # --- End Time Loss Computation ---

            # --- Time Backward Pass ---
            bwd_start = time.perf_counter()
            loss.backward()
            bwd_end = time.perf_counter()
            epoch_backward_s += (bwd_end - bwd_start)
            # --- End Time Backward Pass ---

            # --- Time Optimizer Step ---
            update_start = time.perf_counter()
            optimizer.step()
            update_end = time.perf_counter()
            epoch_update_s += (update_end - update_start)
            # --- End Time Optimizer Step ---

            # --- Metrics (No specific per-batch metric calc needed here) ---
            metric_calc_start = time.perf_counter()
            batch_loss = loss.item()
            running_loss += batch_loss
            metric_calc_end = time.perf_counter()
            epoch_metric_calc_s += (metric_calc_end - metric_calc_start) # Minimal time
            # --- End Time Metric Calc ---

            batches_processed_this_epoch += 1
            total_train_batches_processed += 1

            if (i + 1) % 5 == 0 or (i + 1) == total_train_batches_per_epoch:
                avg_loss = running_loss / batches_processed_this_epoch
                print(f"  Batch {i+1:>4}/{total_train_batches_per_epoch:<5} Loss: {avg_loss:<12.6f}", end='\r')

            batch_iter_start_time = time.perf_counter() # Reset for next iteration

        # End of Epoch
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times_s.append(epoch_duration)
        avg_epoch_loss = running_loss / batches_processed_this_epoch if batches_processed_this_epoch > 0 else 0
        gpu_mem = get_current_gpu_memory_mb() if is_cuda else 0

        print() # Newline after batch progress
        print(f"  Train Loss ({loss_type_str}): {avg_epoch_loss:<12.6f}")
        print(f"  Epoch Time: {epoch_duration:.2f} s")
        if batches_processed_this_epoch > 0:
             print(f"  Avg Batch Timings (ms): "
                   f"Load: {epoch_data_load_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"ToDev: {epoch_to_device_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Fwd: {epoch_forward_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Loss: {epoch_loss_compute_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Bwd: {epoch_backward_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Update: {epoch_update_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Metric: {epoch_metric_calc_s * 1000 / batches_processed_this_epoch:.3f}") # Placeholder
        if is_cuda: print(f"  GPU Mem Used (MB): {gpu_mem:.2f}")

        # Accumulate totals
        total_data_load_s += epoch_data_load_s
        total_to_device_s += epoch_to_device_s
        total_forward_s += epoch_forward_s
        total_loss_compute_s += epoch_loss_compute_s
        total_backward_s += epoch_backward_s
        total_update_s += epoch_update_s
        total_metric_calc_s += epoch_metric_calc_s

    train_end_overall = time.perf_counter()
    total_train_time_s = train_end_overall - train_start_overall
    avg_epoch_time = sum(epoch_times_s) / len(epoch_times_s) if epoch_times_s else 0
    print("-" * 60)
    print(f"Training complete! Total Training Time: {total_train_time_s:.2f} s ({mode_tag}), Avg Epoch Time: {avg_epoch_time:.2f} s")

    # --- Testing Phase ---
    print(f"\n--- Testing {mode_tag} ---")
    network.eval()
    test_loss_sum = 0.0 # Sum of losses for MAE calculation
    test_mse_sum = 0.0 # Sum of squared errors for RMSE
    test_samples = 0
    all_test_preds_denorm_host = []
    all_test_targets_denorm_host = []

    test_start_time = time.perf_counter()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(current_device), targets.to(current_device)
            outputs = network(inputs)

            # Calculate loss used during training (e.g., MSE or MAE)
            loss = criterion(outputs, targets)
            test_loss_sum += loss.item() * inputs.size(0) # Accumulate sum of losses

            # Calculate MSE separately for RMSE reporting
            mse_loss = F.mse_loss(outputs, targets, reduction='sum') # Sum of squared errors
            test_mse_sum += mse_loss.item()

            test_samples += inputs.size(0)

            # Denormalize and store on CPU for final MAE calculation
            preds_denorm = test_dataset.denormalize_targets(outputs).cpu()
            targets_denorm = test_dataset.denormalize_targets(targets).cpu()
            all_test_preds_denorm_host.append(preds_denorm)
            all_test_targets_denorm_host.append(targets_denorm)

    test_end_time = time.perf_counter()
    test_duration_s = test_end_time - test_start_time

    # Concatenate all test results
    all_preds = torch.cat(all_test_preds_denorm_host)
    all_targets = torch.cat(all_test_targets_denorm_host)

    # Calculate final metrics
    avg_test_loss = test_loss_sum / test_samples if test_samples > 0 else 0 # Average loss (MSE or MAE)
    test_rmse = math.sqrt(test_mse_sum / test_samples) if test_samples > 0 else 0
    test_mae = torch.abs(all_preds - all_targets).mean().item() if test_samples > 0 else 0

    print(f"Test Results:")
    print(f"  Average Loss ({loss_type_str}): {avg_test_loss:.6f}")
    print(f"  RMSE (denormalized): {test_rmse:.6f}")
    print(f"  MAE (denormalized): {test_mae:.6f}")
    print(f"  Test Duration: {test_duration_s:.2f} s")

    # Print some sample predictions (denormalized)
    num_samples_to_show = min(10, test_samples)
    if num_samples_to_show > 0:
        print("\n--- Sample Test Predictions (Denormalized) ---")
        print(f"{'Sample':<6} | {'True Value':<15} | {'Predicted Value':<18} | {'Difference':<15}")
        print("-" * 70)
        for i in range(num_samples_to_show):
            true_val = all_targets[i].item()
            pred_val = all_preds[i].item()
            diff = pred_val - true_val
            print(f"{i:<6} | {true_val:<15.4f} | {pred_val:<18.4f} | {diff:<15.4f}")
        print("-" * 70)


    # --- Final Stats ---
    peak_cpu_mem = get_peak_cpu_memory_mb()
    peak_gpu_mem = get_peak_gpu_memory_mb() if is_cuda else 0
    avg_latency_ms = (total_train_time_s * 1000 / total_train_batches_processed) if total_train_batches_processed > 0 else 0
    # Dataset size for throughput might be different if split was used
    throughput_samples_s = (total_train_batches_processed * batch_size / total_train_time_s) if total_train_time_s > 0 else 0

    print("\n--- Final Summary ---")
    print(f"Test Loss ({loss_type_str}):     {avg_test_loss:.6f}")
    print(f"Test MAE (denorm):       {test_mae:.6f}")
    print(f"Test RMSE (denorm):      {test_rmse:.6f}")
    print("--- Performance Metrics ---")
    print(f"Total Train Time (s):    {total_train_time_s:.2f}")
    print(f"Total Test Time (s):     {test_duration_s:.2f}")
    print(f"Avg. Latency/Batch (ms): {avg_latency_ms:.2f}")
    print(f"Throughput (Samples/sec):{throughput_samples_s:.0f}")
    print(f"Avg. Time per Epoch (s): {avg_epoch_time:.2f}")
    if peak_cpu_mem >= 0: print(f"Peak CPU Memory (MB):    {peak_cpu_mem:.2f} (approx)")
    else: print("Peak CPU Memory (MB):    N/A")
    if is_cuda: print(f"Peak GPU Memory (MB):    {peak_gpu_mem:.2f}")

    if total_train_batches_processed > 0:
        print("--- Avg Batch Breakdown (ms) ---")
        print(f"  Data Load: {total_data_load_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  To Device: {total_to_device_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Forward:   {total_forward_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Loss Comp: {total_loss_compute_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Backward:  {total_backward_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Update:    {total_update_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Metric:    {total_metric_calc_s * 1000 / total_train_batches_processed:.3f}") # Placeholder
    print("-" * 25)

    del network, train_loader, test_loader, test_dataset,train_dataset, inputs, targets, outputs, loss
    if is_cuda: torch.cuda.empty_cache()
    gc.collect()


# Mimics california_regression.cpp / cpu_california_regression.cpp
def train_california_regressor(batch_size, num_epochs, dataset_path, learning_rate, loss_type_str, use_cuda_if_available=True):

    # --- Setup ---
    current_device = device if use_cuda_if_available and torch.cuda.is_available() else torch.device("cpu")
    is_cuda = current_device.type == 'cuda'
    mode_tag = "(GPU)" if is_cuda else "(CPU)"
    test_split_ratio = 0.2 # Use 20% for testing
    random_seed = 42 # For reproducible splits

    print(f"\n=== California Housing Regression Training {mode_tag} ===")
    print(f"Using CUDA: {'Yes' if is_cuda else 'No'}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Loss: {loss_type_str}")
    print(f"Dataset Path: {dataset_path}")

    program_start_time = time.perf_counter()
    epoch_times_s = []
    total_train_batches_processed = 0
    # --- Timing Accumulators (Total) ---
    total_data_load_s = 0.0 # Will include preprocessing now
    total_to_device_s = 0.0
    total_forward_s = 0.0
    total_loss_compute_s = 0.0
    total_backward_s = 0.0
    total_update_s = 0.0
    total_metric_calc_s = 0.0 # Placeholder

    # --- Data Loading and Preprocessing ---
    data_load_start = time.perf_counter()
    try:
        print(f"Loading data from: {dataset_path}")
        housing_df = pd.read_csv(dataset_path)

        # Separate features (X) and target (y)
        target_column = 'median_house_value'
        X = housing_df.drop(target_column, axis=1)
        y = housing_df[target_column]

        # Identify feature types
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
        print(f"Numerical features: {numerical_features}")
        print(f"Categorical features: {categorical_features}")

        # Split data BEFORE imputation/scaling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split_ratio, random_state=random_seed
        )
        print(f"Data split: {len(X_train)} train samples, {len(X_test)} test samples.")

        # Create preprocessing pipelines
        # Pipeline for numerical features: Impute missing values (median) then scale
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline for categorical features: Impute missing (most frequent) then one-hot encode
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')) # Ignore categories in test set not seen in train set
        ])

        # Combine pipelines using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ],
            remainder='passthrough' # Keep other columns if any (shouldn't be needed here)
        )

        # Apply preprocessing: fit on train, transform train and test
        print("Applying preprocessing (imputation, scaling, one-hot encoding)...")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        print(f"Processed feature shapes: Train {X_train_processed.shape}, Test {X_test_processed.shape}")

        # Get feature names after one-hot encoding (useful for debugging)
        try:
            feature_names_out = preprocessor.get_feature_names_out()
            num_processed_features = len(feature_names_out)
            # print(f"Processed feature names: {feature_names_out}") # Can be very long
            print(f"Total features after preprocessing: {num_processed_features}")
        except Exception as e_fn:
            print(f"Could not get feature names: {e_fn}")
            num_processed_features = X_train_processed.shape[1]


        # Scale the target variable (y) separately
        target_scaler = StandardScaler()
        # Reshape y_train and y_test to be 2D arrays [n_samples, 1] for scaler
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

        # Extract target mean and std dev for denormalization later
        target_mean_ = target_scaler.mean_[0] # Get scalar value
        target_std_ = target_scaler.scale_[0] # Get scalar value (std dev)

        # Create Dataset instances with PREPROCESSED data
        train_dataset = CaliforniaDataset(X_train_processed, y_train_scaled, target_mean_, target_std_)
        test_dataset = CaliforniaDataset(X_test_processed, y_test_scaled, target_mean_, target_std_)

        # Get the actual number of features after preprocessing
        num_features = train_dataset.get_num_features()
        output_size = 1 # Regression target

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error during data loading or preprocessing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    data_load_end = time.perf_counter()
    total_data_load_s += (data_load_end - data_load_start)
    print(f"Data loading and preprocessing took: {total_data_load_s:.2f} s")

    num_workers = 4 if is_cuda else 0
    pin_memory = is_cuda
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    total_train_batches_per_epoch = len(train_loader)
    total_test_batches = len(test_loader)
    print(f"Train batches/epoch: {total_train_batches_per_epoch}, Test batches: {total_test_batches}")

    # --- Model, Optimizer, Loss ---
    try:
        network = RegressionModel(num_features=num_features, output_size=output_size).to(current_device)
        if not list(network.parameters()):
             raise RuntimeError("Network has no parameters.")
    except Exception as e:
        print(f"Error creating network: {e}", file=sys.stderr)
        return

    optimizer = optim.Adam(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    if loss_type_str.lower() == "mse": criterion = nn.MSELoss()
    elif loss_type_str.lower() == "mae": criterion = nn.L1Loss()
    elif loss_type_str.lower() == "custom": criterion = custom_loss_function
    else:
        print(f"Warning: Unsupported loss type '{loss_type_str}'. Defaulting to MSE.", file=sys.stderr)
        criterion = nn.MSELoss(); loss_type_str = "MSE"

    print(f"Using Optimizer: Adam (LR={learning_rate})")
    print(f"Using Loss: {loss_type_str}")

    # --- Training Loop ---
    print("Starting training...")
    # print(f"{'Epoch':<6} | {'Batch':<10} | {'Loss':<12} | {'Time/Epoch (s)':<15} | {'GPU Mem (MB)':<12}")
    # print("-" * 60)

    train_start_overall = time.perf_counter()
    if is_cuda: torch.cuda.reset_peak_memory_stats(current_device)

    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        network.train()
        running_loss = 0.0
        batches_processed_this_epoch = 0
        # --- Timing Accumulators (Epoch) ---
        epoch_data_load_s = 0.0
        epoch_to_device_s = 0.0
        epoch_forward_s = 0.0
        epoch_loss_compute_s = 0.0
        epoch_backward_s = 0.0
        epoch_update_s = 0.0
        epoch_metric_calc_s = 0.0 # Placeholder

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        batch_iter_start_time = time.perf_counter()

        for i, (inputs, targets) in enumerate(train_loader):
            batch_iter_end_time = time.perf_counter()
            epoch_data_load_s += (batch_iter_end_time - batch_iter_start_time)

            # --- Time To Device ---
            to_device_start = time.perf_counter()
            inputs, targets = inputs.to(current_device), targets.to(current_device)
            to_device_end = time.perf_counter()
            epoch_to_device_s += (to_device_end - to_device_start)
            # --- End Time To Device ---

            optimizer.zero_grad()

            # --- Time Forward Pass ---
            fwd_start = time.perf_counter()
            outputs = network(inputs)
            fwd_end = time.perf_counter()
            epoch_forward_s += (fwd_end - fwd_start)
            # --- End Time Forward Pass ---

            # --- Time Loss Computation ---
            loss_start = time.perf_counter()
            loss = criterion(outputs, targets)
            loss_end = time.perf_counter()
            epoch_loss_compute_s += (loss_end - loss_start)
            # --- End Time Loss Computation ---

            # --- Time Backward Pass ---
            bwd_start = time.perf_counter()
            loss.backward()
            bwd_end = time.perf_counter()
            epoch_backward_s += (bwd_end - bwd_start)
            # --- End Time Backward Pass ---

            # --- Time Optimizer Step ---
            update_start = time.perf_counter()
            optimizer.step()
            update_end = time.perf_counter()
            epoch_update_s += (update_end - update_start)
            # --- End Time Optimizer Step ---

            # --- Metrics ---
            metric_calc_start = time.perf_counter()
            batch_loss = loss.item()
            running_loss += batch_loss
            metric_calc_end = time.perf_counter()
            epoch_metric_calc_s += (metric_calc_end - metric_calc_start)
            # --- End Time Metric Calc ---

            batches_processed_this_epoch += 1
            total_train_batches_processed += 1

            if (i + 1) % 50 == 0 or (i + 1) == total_train_batches_per_epoch:
                avg_loss = running_loss / batches_processed_this_epoch
                print(f"  Batch {i+1:>4}/{total_train_batches_per_epoch:<5} Loss: {avg_loss:<12.6f}", end='\r')

            batch_iter_start_time = time.perf_counter() # Reset for next iteration

        # End of Epoch
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times_s.append(epoch_duration)
        avg_epoch_loss = running_loss / batches_processed_this_epoch if batches_processed_this_epoch > 0 else 0
        gpu_mem = get_current_gpu_memory_mb() if is_cuda else 0

        print() # Newline after batch progress
        print(f"  Train Loss ({loss_type_str}): {avg_epoch_loss:<12.6f}")
        print(f"  Epoch Time: {epoch_duration:.2f} s")
        if batches_processed_this_epoch > 0:
             print(f"  Avg Batch Timings (ms): "
                   f"Load: {epoch_data_load_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"ToDev: {epoch_to_device_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Fwd: {epoch_forward_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Loss: {epoch_loss_compute_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Bwd: {epoch_backward_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Update: {epoch_update_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Metric: {epoch_metric_calc_s * 1000 / batches_processed_this_epoch:.3f}") # Placeholder
        if is_cuda: print(f"  GPU Mem Used (MB): {gpu_mem:.2f}")

        # Accumulate totals
        total_data_load_s += epoch_data_load_s
        total_to_device_s += epoch_to_device_s
        total_forward_s += epoch_forward_s
        total_loss_compute_s += epoch_loss_compute_s
        total_backward_s += epoch_backward_s
        total_update_s += epoch_update_s
        total_metric_calc_s += epoch_metric_calc_s

    train_end_overall = time.perf_counter()
    total_train_time_s = train_end_overall - train_start_overall
    avg_epoch_time = sum(epoch_times_s) / len(epoch_times_s) if epoch_times_s else 0
    print("-" * 60)
    print(f"Training complete! Total Training Time: {total_train_time_s:.2f} s ({mode_tag}), Avg Epoch Time: {avg_epoch_time:.2f} s")

    # --- Testing Phase ---
    print(f"\n--- Testing {mode_tag} ---")
    network.eval()
    test_loss_sum = 0.0
    test_mse_sum = 0.0
    test_samples = 0
    all_test_preds_denorm_host = []
    all_test_targets_denorm_host = []

    test_start_time = time.perf_counter()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(current_device), targets.to(current_device)
            outputs = network(inputs)

            loss = criterion(outputs, targets)
            test_loss_sum += loss.item() * inputs.size(0)

            mse_loss = F.mse_loss(outputs, targets, reduction='sum')
            test_mse_sum += mse_loss.item()

            test_samples += inputs.size(0)

            preds_denorm = test_dataset.denormalize_targets(outputs).cpu()
            targets_denorm = test_dataset.denormalize_targets(targets).cpu()
            all_test_preds_denorm_host.append(preds_denorm)
            all_test_targets_denorm_host.append(targets_denorm)

    test_end_time = time.perf_counter()
    test_duration_s = test_end_time - test_start_time

    all_preds = torch.cat(all_test_preds_denorm_host)
    all_targets = torch.cat(all_test_targets_denorm_host)

    avg_test_loss = test_loss_sum / test_samples if test_samples > 0 else 0
    test_rmse = math.sqrt(test_mse_sum / test_samples) if test_samples > 0 else 0
    test_mae = torch.abs(all_preds - all_targets).mean().item() if test_samples > 0 else 0

    print(f"Test Results:")
    print(f"  Average Loss ({loss_type_str}): {avg_test_loss:.6f}")
    print(f"  RMSE (denormalized): {test_rmse:.6f}")
    print(f"  MAE (denormalized): {test_mae:.6f}")
    print(f"  Test Duration: {test_duration_s:.2f} s")

    num_samples_to_show = min(10, test_samples)
    if num_samples_to_show > 0:
        print("\n--- Sample Test Predictions (Denormalized) ---")
        print(f"{'Sample':<6} | {'True Value':<15} | {'Predicted Value':<18} | {'Difference':<15}")
        print("-" * 70)
        for i in range(num_samples_to_show):
            true_val = all_targets[i].item()
            pred_val = all_preds[i].item()
            diff = pred_val - true_val
            # California values can be large, adjust formatting
            print(f"{i:<6} | {true_val:<15.2f} | {pred_val:<18.2f} | {diff:<15.2f}")
        print("-" * 70)

    # --- Final Stats ---
    peak_cpu_mem = get_peak_cpu_memory_mb()
    peak_gpu_mem = get_peak_gpu_memory_mb() if is_cuda else 0
    avg_latency_ms = (total_train_time_s * 1000 / total_train_batches_processed) if total_train_batches_processed > 0 else 0
    throughput_samples_s = (total_train_batches_processed * batch_size / total_train_time_s) if total_train_time_s > 0 else 0

    print("\n--- Final Summary ---")
    print(f"Test Loss ({loss_type_str}):     {avg_test_loss:.6f}")
    print(f"Test MAE (denorm):       {test_mae:.6f}")
    print(f"Test RMSE (denorm):      {test_rmse:.6f}")
    print("--- Performance Metrics ---")
    print(f"Total Train Time (s):    {total_train_time_s:.2f}")
    print(f"Avg. Latency/Batch (ms): {avg_latency_ms:.2f}")
    print(f"Throughput (Samples/sec):{throughput_samples_s:.0f}")
    print(f"Avg. Time per Epoch (s): {avg_epoch_time:.2f}")
    if peak_cpu_mem >= 0: print(f"Peak CPU Memory (MB):    {peak_cpu_mem:.2f} (approx)")
    else: print("Peak CPU Memory (MB):    N/A")
    if is_cuda: print(f"Peak GPU Memory (MB):    {peak_gpu_mem:.2f}")

    if total_train_batches_processed > 0:
        print("--- Avg Batch Breakdown (ms) ---")
        print(f"  Data Load: {total_data_load_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  To Device: {total_to_device_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Forward:   {total_forward_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Loss Comp: {total_loss_compute_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Backward:  {total_backward_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Update:    {total_update_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Metric:    {total_metric_calc_s * 1000 / total_train_batches_processed:.3f}") # Placeholder
    print("-" * 25)

    del network, train_loader, test_loader, test_dataset,train_dataset,inputs, targets, outputs, loss
    if is_cuda: torch.cuda.empty_cache()
    gc.collect()


# --- Autoencoder Display Function (Mimics display_ae_samples) ---
def display_ae_samples_opencv(inputs_host, outputs_host, height, width, use_grayscale, loss_type, num_samples=5):
    """Displays original and reconstructed images using OpenCV and terminal."""
    if inputs_host is None or outputs_host is None:
        print("Warning: Cannot display AE samples due to missing host data.", file=sys.stderr)
        return

    batch_size = inputs_host.size(0)
    samples_to_show = min(num_samples, batch_size)
    channels = inputs_host.size(1)

    print(f"\n=== Validation Samples (Autoencoder - Showing up to {samples_to_show}) ===")
    print(f"{'Sample':<6} | {'Loss':<12} | {'Input (Terminal)':<35} | {'Output (Terminal)':<35}")
    print("-" * 95)

    # Terminal display characters
    shades = [" ", "░", "▒", "▓", "█"]
    max_term_height = 15
    max_term_width = 30

    for i in range(samples_to_show):
        input_img_tensor = inputs_host[i] # Shape (C, H, W)
        output_img_tensor = outputs_host[i] # Shape (C, H, W)

        # Calculate sample loss (on CPU)
        # Use MSE for AE loss calculation regardless of training loss for display simplicity?
        # Or use the provided loss_type
        if loss_type.lower() == "mse":
             sample_loss = F.mse_loss(output_img_tensor, input_img_tensor).item()
        elif loss_type.lower() == "custom":
             # Custom loss needs element-wise application or careful batching
             # For simplicity, let's use MSE for display loss here
             sample_loss = F.mse_loss(output_img_tensor, input_img_tensor).item()
             # sample_loss = calculate_custom_loss_cpu(output_img_tensor.flatten().tolist(), input_img_tensor.flatten().tolist()) # If needed
        else: # Default to MSE
             sample_loss = F.mse_loss(output_img_tensor, input_img_tensor).item()


        # --- Prepare images for OpenCV and Terminal ---
        def prepare_display(img_tensor, title):
            img_np = img_tensor.numpy() # Shape (C, H, W)
            if channels > 1:
                img_np = np.transpose(img_np, (1, 2, 0)) # Convert to (H, W, C) for OpenCV
            else:
                img_np = img_np.squeeze(0) # Remove channel dim -> (H, W)

            # Clamp values to [0, 1] and convert to uint8 [0, 255]
            img_np = np.clip(img_np, 0, 1)
            img_8u = (img_np * 255).astype(np.uint8)

            # --- OpenCV Window ---
            try:
                display_window_width = 256
                display_window_height = 256
                resized_for_display = cv2.resize(img_8u, (display_window_width, display_window_height), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(title, resized_for_display)
            except Exception as e:
                # print(f"Warning: OpenCV display failed for '{title}'. Error: {e}", file=sys.stderr)
                # Silently ignore if GUI is not available
                pass


            # --- Terminal Representation ---
            term_h = min(height, max_term_height)
            term_w = min(width, max_term_width)

            # Ensure grayscale for terminal
            if len(img_8u.shape) == 3 and img_8u.shape[2] == 3: # HWC
                gray_img = cv2.cvtColor(img_8u, cv2.COLOR_BGR2GRAY)
            elif len(img_8u.shape) == 2: # HW
                gray_img = img_8u
            else: # Unexpected shape
                gray_img = np.zeros((height, width), dtype=np.uint8)

            resized_term = cv2.resize(gray_img, (term_w, term_h), interpolation=cv2.INTER_NEAREST)

            terminal_lines = []
            for r in range(term_h):
                line = ""
                for c in range(term_w):
                    pixel_value = resized_term[r, c]
                    shade_index = min(int(pixel_value / 256 * len(shades)), len(shades) - 1)
                    line += shades[shade_index] * 2 # Double chars for better aspect ratio
                terminal_lines.append(line)
            return terminal_lines

        input_term_lines = prepare_display(input_img_tensor, f"Input {i}")
        output_term_lines = prepare_display(output_img_tensor, f"Output {i}")

        # Print side-by-side terminal views
        max_lines = max(len(input_term_lines), len(output_term_lines))
        term_width = max(len(l) for l in input_term_lines + output_term_lines) if input_term_lines or output_term_lines else 0

        print(f"{i:<6} | {sample_loss:<12.6f} | {input_term_lines[0]:<{term_width}} | {output_term_lines[0]:<{term_width}}")
        for line_idx in range(1, max_lines):
            in_line = input_term_lines[line_idx] if line_idx < len(input_term_lines) else ""
            out_line = output_term_lines[line_idx] if line_idx < len(output_term_lines) else ""
            print(f"{'':<6} | {'':<12} | {in_line:<{term_width}} | {out_line:<{term_width}}")
        if i < samples_to_show - 1:
             print("-" * 95) # Separator between samples

    print("-" * 95)
    try:
        print("Press any key in an OpenCV window to continue...")
        cv2.waitKey(0) # Wait indefinitely until a key is pressed in an OpenCV window
        cv2.destroyAllWindows() # Close windows afterwards
    except Exception:
        # print("Warning: Failed to wait for key press (maybe no GUI?).", file=sys.stderr)
        pass # Ignore if no GUI


# Mimics conv.cpp / cpu_conv.cpp (Autoencoder Training)
def train_conv_autoencoder(batch_size, num_epochs, dataset_path, image_height, image_width, use_grayscale, learning_rate=0.001, loss_type_str="mse", use_cuda_if_available=True):

    # --- Setup ---
    current_device = device if use_cuda_if_available and torch.cuda.is_available() else torch.device("cpu")
    is_cuda = current_device.type == 'cuda'
    mode_tag = "(GPU)" if is_cuda else "(CPU)"

    print(f"\n=== Convolutional Autoencoder Training {mode_tag} ===")
    print(f"Using CUDA: {'Yes' if is_cuda else 'No'}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Loss: {loss_type_str}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Image Size: {image_width}x{image_height}, Grayscale: {use_grayscale}")

    program_start_time = time.perf_counter()
    epoch_times_s = []
    total_train_batches_processed = 0
    input_channels = 1 if use_grayscale else 3
    # --- Timing Accumulators (Total) ---
    total_data_load_s = 0.0
    total_to_device_s = 0.0
    total_forward_s = 0.0
    total_loss_compute_s = 0.0
    total_backward_s = 0.0
    total_update_s = 0.0
    total_metric_calc_s = 0.0 # Placeholder

    # --- Data ---
    try:
        print("Loading dataset...")
        load_start_time = time.perf_counter()
        # Use ButterflyDataset for AE as per C++ conv.cpp
        # Need CSV path and root dir
        # Assuming dataset_path is the root dir containing train/test subfolders and CSVs
        root_dir = Path(dataset_path)
        train_csv = root_dir / "Training_set.csv" # Adjust if CSVs are elsewhere
        test_csv = root_dir / "Testing_set.csv"   # Adjust if CSVs are elsewhere

        # Use ButterflyDataset, which loads images based on CSV
        train_dataset = ButterflyDataset(str(train_csv), str(root_dir), width=image_width, height=image_height, rgb=not use_grayscale)
        # Use test set for validation display
        val_dataset = ButterflyDataset(str(test_csv), str(root_dir), width=image_width, height=image_height, rgb=not use_grayscale)

        load_end_time = time.perf_counter()
        print(f"  Dataset loading time: {load_end_time - load_start_time:.2f} s")
        if len(train_dataset) == 0 or len(val_dataset) == 0:
             raise RuntimeError("Loaded 0 samples for train or validation set.")
    except Exception as e:
        print(f"Error loading Butterfly dataset for AE: {e}", file=sys.stderr)
        return

    num_workers = 4 if is_cuda else 0
    pin_memory = is_cuda
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    total_train_batches_per_epoch = len(train_loader)
    total_val_batches = len(val_loader)

    # --- Model, Optimizer, Loss ---
    try:
        # Use the ConvAutoencoder model defined earlier
        network = ConvAutoencoder(input_channels=input_channels, input_height=image_height, input_width=image_width).to(current_device)
        if not list(network.parameters()):
             raise RuntimeError("Network has no parameters.")
    except Exception as e:
        print(f"Error creating network: {e}", file=sys.stderr)
        return

    optimizer = optim.Adam(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    if loss_type_str.lower() == "mse": criterion = nn.MSELoss()
    elif loss_type_str.lower() == "custom": criterion = custom_loss_function
    else:
        print(f"Warning: Unsupported loss type '{loss_type_str}'. Defaulting to MSE.", file=sys.stderr)
        criterion = nn.MSELoss(); loss_type_str = "MSE"

    print(f"Using Optimizer: Adam (LR={learning_rate})")
    print(f"Using Loss: {loss_type_str}")

    # --- Training Loop ---
    print("Starting training...")
    # print(f"{'Epoch':<6} | {'Batch':<10} | {'Loss':<12} | {'Time/Epoch (s)':<15} | {'GPU Mem (MB)':<12}")
    # print("-" * 60)

    train_start_overall = time.perf_counter()
    if is_cuda: torch.cuda.reset_peak_memory_stats(current_device)

    first_batch_inputs_h = None
    first_batch_outputs_h = None
    val_start_time = 0 # Initialize
    val_duration_s = 0 # Initialize

    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        network.train()
        running_loss = 0.0
        batches_processed_this_epoch = 0
        # --- Timing Accumulators (Epoch) ---
        epoch_data_load_s = 0.0
        epoch_to_device_s = 0.0
        epoch_forward_s = 0.0
        epoch_loss_compute_s = 0.0
        epoch_backward_s = 0.0
        epoch_update_s = 0.0
        epoch_metric_calc_s = 0.0 # Placeholder

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        batch_iter_start_time = time.perf_counter()

        # ButterflyDataset returns (image, label_idx), we only need image for AE
        for i, (inputs, _) in enumerate(train_loader):
            batch_iter_end_time = time.perf_counter()
            epoch_data_load_s += (batch_iter_end_time - batch_iter_start_time)

            # --- Time To Device ---
            to_device_start = time.perf_counter()
            inputs = inputs.to(current_device)
            targets = inputs # Target is the input itself for AE
            to_device_end = time.perf_counter()
            epoch_to_device_s += (to_device_end - to_device_start)
            # --- End Time To Device ---

            optimizer.zero_grad()

            # --- Time Forward Pass ---
            fwd_start = time.perf_counter()
            outputs = network(inputs)
            fwd_end = time.perf_counter()
            epoch_forward_s += (fwd_end - fwd_start)
            # --- End Time Forward Pass ---

            # --- Time Loss Computation ---
            loss_start = time.perf_counter()
            loss = criterion(outputs, targets)
            loss_end = time.perf_counter()
            epoch_loss_compute_s += (loss_end - loss_start)
            # --- End Time Loss Computation ---

            # --- Time Backward Pass ---
            bwd_start = time.perf_counter()
            loss.backward()
            bwd_end = time.perf_counter()
            epoch_backward_s += (bwd_end - bwd_start)
            # --- End Time Backward Pass ---

            # --- Time Optimizer Step ---
            update_start = time.perf_counter()
            optimizer.step()
            update_end = time.perf_counter()
            epoch_update_s += (update_end - update_start)
            # --- End Time Optimizer Step ---

            # --- Metrics ---
            metric_calc_start = time.perf_counter()
            batch_loss = loss.item()
            running_loss += batch_loss
            metric_calc_end = time.perf_counter()
            epoch_metric_calc_s += (metric_calc_end - metric_calc_start)
            # --- End Time Metric Calc ---

            batches_processed_this_epoch += 1
            total_train_batches_processed += 1

            if (i + 1) % 50 == 0 or (i + 1) == total_train_batches_per_epoch:
                avg_loss = running_loss / batches_processed_this_epoch
                print(f"  Batch {i+1:>4}/{total_train_batches_per_epoch:<5} Loss: {avg_loss:<12.6f}", end='\r')

            batch_iter_start_time = time.perf_counter() # Reset for next iteration

        # End of Epoch
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times_s.append(epoch_duration)
        avg_epoch_loss = running_loss / batches_processed_this_epoch if batches_processed_this_epoch > 0 else 0
        gpu_mem = get_current_gpu_memory_mb() if is_cuda else 0

        print() # Newline after batch progress
        print(f"  Train Loss ({loss_type_str}): {avg_epoch_loss:<12.6f}")
        print(f"  Epoch Time: {epoch_duration:.2f} s")
        if batches_processed_this_epoch > 0:
             print(f"  Avg Batch Timings (ms): "
                   f"Load: {epoch_data_load_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"ToDev: {epoch_to_device_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Fwd: {epoch_forward_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Loss: {epoch_loss_compute_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Bwd: {epoch_backward_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Update: {epoch_update_s * 1000 / batches_processed_this_epoch:.3f} | "
                   f"Metric: {epoch_metric_calc_s * 1000 / batches_processed_this_epoch:.3f}") # Placeholder
        if is_cuda: print(f"  GPU Mem Used (MB): {gpu_mem:.2f}")

        # Accumulate totals
        total_data_load_s += epoch_data_load_s
        total_to_device_s += epoch_to_device_s
        total_forward_s += epoch_forward_s
        total_loss_compute_s += epoch_loss_compute_s
        total_backward_s += epoch_backward_s
        total_update_s += epoch_update_s
        total_metric_calc_s += epoch_metric_calc_s

        # --- Validation Phase (Only at end of training) ---
        if epoch == num_epochs - 1:
            print(f"\n--- Validation {mode_tag} ---")
            network.eval()
            val_loss = 0.0
            val_batches_processed = 0
            val_start_time = time.perf_counter()
            with torch.no_grad():
                for i, (inputs, _) in enumerate(val_loader):
                    inputs = inputs.to(current_device)
                    targets = inputs # AE target is input
                    outputs = network(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_batches_processed += 1

                    if i == 0: # Capture first batch for display
                        first_batch_inputs_h = inputs.cpu()
                        first_batch_outputs_h = outputs.cpu()

            val_end_time = time.perf_counter()
            val_duration_s = val_end_time - val_start_time
            avg_val_loss = val_loss / val_batches_processed if val_batches_processed > 0 else 0
            print(f"Validation Loss ({loss_type_str}): {avg_val_loss:.6f}")
            print(f"Validation Duration: {val_duration_s:.2f} s")

            if first_batch_inputs_h is not None and first_batch_outputs_h is not None:
                display_ae_samples_opencv(
                    inputs_host=first_batch_inputs_h,
                    outputs_host=first_batch_outputs_h,
                    height=image_height,
                    width=image_width,
                    use_grayscale=use_grayscale,
                    loss_type=loss_type_str,
                    num_samples=5
                )
            else:
                print("No validation batch data captured for display.")


    train_end_overall = time.perf_counter()
    total_train_time_s = train_end_overall - train_start_overall
    avg_epoch_time = sum(epoch_times_s) / len(epoch_times_s) if epoch_times_s else 0
    print("-" * 60)
    print(f"Training complete! Total Training Time: {total_train_time_s:.2f} s ({mode_tag}), Avg Epoch Time: {avg_epoch_time:.2f} s")

    # --- Final Stats ---
    peak_cpu_mem = get_peak_cpu_memory_mb()
    peak_gpu_mem = get_peak_gpu_memory_mb() if is_cuda else 0
    avg_latency_ms = (total_train_time_s * 1000 / total_train_batches_processed) if total_train_batches_processed > 0 else 0
    throughput_samples_s = (total_train_batches_processed * batch_size / total_train_time_s) if total_train_time_s > 0 else 0

    print("\n--- Final Summary ---")
    print(f"Validation Loss ({loss_type_str}): {avg_val_loss:.6f}")
    print("--- Performance Metrics ---")
    print(f"Total Train Time (s):    {total_train_time_s:.2f}")
    print(f"Total Val Time (s):      {val_duration_s:.2f}")
    print(f"Avg. Latency/Batch (ms): {avg_latency_ms:.2f}")
    print(f"Throughput (Samples/sec):{throughput_samples_s:.0f}")
    print(f"Avg. Time per Epoch (s): {avg_epoch_time:.2f}")
    if peak_cpu_mem >= 0: print(f"Peak CPU Memory (MB):    {peak_cpu_mem:.2f} (approx)")
    else: print("Peak CPU Memory (MB):    N/A")
    if is_cuda: print(f"Peak GPU Memory (MB):    {peak_gpu_mem:.2f}")

    if total_train_batches_processed > 0:
        print("--- Avg Batch Breakdown (ms) ---")
        print(f"  Data Load: {total_data_load_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  To Device: {total_to_device_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Forward:   {total_forward_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Loss Comp: {total_loss_compute_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Backward:  {total_backward_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Update:    {total_update_s * 1000 / total_train_batches_processed:.3f}")
        print(f"  Metric:    {total_metric_calc_s * 1000 / total_train_batches_processed:.3f}") # Placeholder
    print("-" * 25)

    del network, train_loader, val_loader, train_dataset, inputs, targets, outputs, loss
    if is_cuda: torch.cuda.empty_cache()
    gc.collect()


# --- Main Execution Block (Mimics main.cpp) ---
if __name__ == "__main__":

    # Mimic debug level setting (less direct effect)
    # set_debug_level(LEVEL_WARN)
    print(f"Current debug level (conceptual): {current_debug_level}")

    # Mimic color printing
    print("\033[1;31mRed Text\033[0m")
    print("\033[1;32mGreen Text\033[0m")
    print("\033[1;34mBlue Text\033[0m")

    # --- Parameters (adjust paths as needed) ---
    # It's better to use argparse for flexibility
    parser = argparse.ArgumentParser(description="PyTorch Training Equivalents")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--run_cpu', action='store_true', help='Force CPU execution')
    # Add specific dataset paths - USE ABSOLUTE PATHS or adjust relative paths
    parser.add_argument('--mnist_path', type=str, default="/home/metarou/Dokumentumok/Programs/NN test/Dataset/", help='Path containing MNIST-JPG folders')
    parser.add_argument('--cifar_path', type=str, default="/home/metarou/Dokumentumok/Programs/NN test/Dataset/CIFAR-10/", help='Path for CIFAR-10 data (will download if needed)')
    parser.add_argument('--boston_csv', type=str, default="/home/metarou/Dokumentumok/Programs/NN test/Dataset/Boston.csv", help='Path to Boston housing CSV')
    parser.add_argument('--california_csv', type=str, default="/home/metarou/Dokumentumok/Programs/NN test/Dataset/housing.csv", help='Path to California housing CSV')
    parser.add_argument('--ae_image_path', type=str, default="/home/metarou/Dokumentumok/Programs/NN test/Dataset", help='Path to images for Autoencoder training')

    args = parser.parse_args()

    use_cuda = not args.run_cpu # Use CUDA unless --run_cpu is specified

    # --- Run Training Functions ---

    # --- Boston Housing Regression ---
    print("\n" + "="*10 + " Starting Boston Housing Regression " + "="*10)
    try:
        train_boston_regressor(
            batch_size=256,
            num_epochs=args.epochs,
            dataset_path=args.boston_csv,
            learning_rate=args.lr,
            loss_type_str="mse",
            use_cuda_if_available=use_cuda
        )
    except Exception as e:
        print(f"Error during Boston Regression training: {e}", file=sys.stderr)
    print("\n" + "="*10 + " Finished Boston Housing Regression " + "="*10)


    # --- California Housing Regression ---
    print("\n" + "="*10 + " Starting California Housing Regression " + "="*10)
    try:
        train_california_regressor(
            batch_size=256,
            num_epochs=args.epochs,
            dataset_path=args.california_csv,
            learning_rate=args.lr,
            loss_type_str="mse",
            use_cuda_if_available=use_cuda
        )
    except Exception as e:
        print(f"Error during California Regression training: {e}", file=sys.stderr)
    print("\n" + "="*10 + " Finished California Housing Regression " + "="*10)


    # --- CIFAR-10 Classification ---
    # print("\n" + "="*10 + " Starting CIFAR-10 Classification " + "="*10)
    # try:
    #     train_cifar_classifier(
    #         batch_size=256,
    #         num_epochs=args.epochs,
    #         dataset_path=args.cifar_path,
    #         image_height=32, image_width=32, # Standard CIFAR size
    #         use_grayscale=False, # CIFAR is color
    #         learning_rate=args.lr,
    #         use_cuda_if_available=use_cuda
    #     )
    # except Exception as e:
    #     print(f"Error during CIFAR-10 training: {e}", file=sys.stderr)
    # print("\n" + "="*10 + " Finished CIFAR-10 Classification " + "="*10)


    # --- MNIST Classification ---
    # print("\n" + "="*10 + " Starting MNIST Classification " + "="*10)
    # try:
    #     train_mnist_classifier(
    #         batch_size=args.batch_size,
    #         num_epochs=args.epochs,
    #         dataset_path=args.mnist_path,
    #         image_height=28, image_width=28, # Standard MNIST size
    #         use_grayscale=True, # MNIST is grayscale
    #         learning_rate=args.lr,
    #         use_cuda_if_available=use_cuda
    #     )
    # except Exception as e:
    #     print(f"Error during MNIST training: {e}", file=sys.stderr)
    # print("\n" + "="*10 + " Finished MNIST Classification " + "="*10)


    # --- Convolutional Autoencoder ---
    print("\n" + "="*10 + " Starting Convolutional Autoencoder " + "="*10)
    try:
        # Parameters from C++ main.cpp conv::train_conv_model call
        ae_image_height = 28 # Example, adjust as needed
        ae_image_width = 28  # Example, adjust as needed
        ae_use_grayscale = True # Example, adjust as needed
        train_conv_autoencoder(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            dataset_path=args.ae_image_path,
            image_height=ae_image_height,
            image_width=ae_image_width,
            use_grayscale=ae_use_grayscale,
            learning_rate=args.lr,
            loss_type_str="mse", # AE typically uses MSE
            use_cuda_if_available=use_cuda
        )
    except Exception as e:
        print(f"Error during Autoencoder training: {e}", file=sys.stderr)
    print("\n" + "="*10 + " Finished Convolutional Autoencoder " + "="*10)


    print("\nAll specified training runs finished.")