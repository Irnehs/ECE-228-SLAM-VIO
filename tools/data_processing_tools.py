import pandas as pd
import os
import requests
import zipfile
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from lightning.pytorch import LightningDataModule
import random
import torch

class IMUImageDataset(Dataset):
    def __init__(self, csv_path, cam0_image_root, cam1_image_root, vio_predictions_path, transform=None, seq_len=5, prediction_len=5, H=224, W=224):
        self.data = pd.read_csv(csv_path)
        self.cam0_image_root = cam0_image_root
        self.cam1_image_root = cam1_image_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor()
        ])
        self.seq_len = seq_len
        self.prediction_len = prediction_len
        self.vio_predictions_path = vio_predictions_path
        self.data['timestamp'] = self.data['timestamp'].astype(float) / 1e9

        self.vio_data = pd.read_csv(self.vio_predictions_path, sep=' ', names=['timestamp', 'pred_p_x', 'pred_p_y', 'pred_p_z', 'pred_q_w', 'pred_q_x', 'pred_q_y', 'pred_q_z'])
        self.data = pd.merge_asof(
            self.data.sort_values('timestamp'),
            self.vio_data.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            suffixes=('_vio', '_gt'),
            tolerance=0.000001 # 100 us tolerance
        )
        self.data = self.data.dropna()
        print(f"Loaded dataset of size {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        cam0_list = []
        cam1_list = []
        imu_list = []
        ground_truth_list = []
        vio_prediction_list = []
        timestamp_list = []

        for i in range(self.seq_len):
            # Load images
            cam0_path = os.path.join(self.cam0_image_root, row['filename_cam0'])
            cam0_image = Image.open(cam0_path)
            cam0_image = self.transform(cam0_image)
            cam0_list.append(cam0_image)

            cam1_path = os.path.join(self.cam1_image_root, row['filename_cam1'])
            cam1_image = Image.open(cam1_path)
            cam1_image = self.transform(cam1_image)
            cam1_list.append(cam1_image)

            # Load IMU
            imu = row[["w_x", "w_y", "w_z", "a_x", "a_y", "a_z"]].values.astype("float32")
            imu_tensor = torch.tensor(imu)
            imu_list.append(imu_tensor)

            # Load timestamp in seconds
            timestamp = float(row["timestamp"])
            timestamp_list.append(torch.tensor(timestamp, dtype=torch.float32))

        # Pose sequence (still returns repeated values per sample for now)
        for i in range(self.prediction_len):
            gt = row[["p_x", "p_y", "p_z", "q_x", "q_y", "q_z", "q_w"]].values.astype("float32")
            vio = row[["pred_p_x", "pred_p_y", "pred_p_z", "pred_q_x", "pred_q_y", "pred_q_z", "pred_q_w"]].values.astype("float32")
            ground_truth_list.append(torch.tensor(gt))
            vio_prediction_list.append(torch.tensor(vio))

        cam0_tensor = torch.stack(cam0_list)  # (seq_len, 3, H, W)
        cam1_tensor = torch.stack(cam1_list)
        imu_tensor = torch.stack(imu_list)    # (seq_len, 6)
        timestamp_tensor = torch.stack(timestamp_list)  # (seq_len,)
        gt_tensor = torch.stack(ground_truth_list)      # (prediction_len, 7)
        vio_tensor = torch.stack(vio_prediction_list)   # (prediction_len, 7)

        return {
            "data": [imu_tensor, cam0_tensor, cam1_tensor],
            "timestamp": timestamp_tensor
        }, {
            "ground_truth": gt_tensor,
            "vio": vio_tensor
        }
def download_dataset(name, url):
    data_dir = os.path.join("data", name)
    zip_path = os.path.join(data_dir, f"{name}.zip")

    print(f"Setting up dataset: {name}")
    os.makedirs(data_dir, exist_ok=True)

    # Streamed download with progress bar
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=name, ncols=80
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    # Unzip into the subdirectory
    print(f"Unzipping into {data_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Remove the zip file
    print("Removing archive...")
    os.remove(zip_path)

    print(f"✅ Dataset '{name}' ready in {data_dir}")


def prep_combined_csv(input_dir, vio_csv_path, output_filepath):
    def load_and_clean_csv(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df.rename(
            columns={
                "#timestamp [ns]": "timestamp",
                "#timestamp": "timestamp",
                "p_RS_R_x [m]": "p_x",
                "p_RS_R_y [m]": "p_y",
                "p_RS_R_z [m]": "p_z",
                "q_RS_x []": "q_x",
                "q_RS_y []": "q_y",
                "q_RS_z []": "q_z",
                "q_RS_w []": "q_w",
                "w_RS_S_x [rad s^-1]": "w_x",
                "w_RS_S_y [rad s^-1]": "w_y",
                "w_RS_S_z [rad s^-1]": "w_z",
                "a_RS_S_x [m s^-2]": "a_x",
                "a_RS_S_y [m s^-2]": "a_y",
                "a_RS_S_z [m s^-2]": "a_z",
            },
            inplace=True,
        )
        df["timestamp"] = df["timestamp"].astype(np.int64)
        df.sort_values("timestamp", inplace=True)
        return df

    # Load and clean
    cam0_df = load_and_clean_csv(os.path.join(input_dir, "cam0", "data.csv"))
    cam1_df = load_and_clean_csv(os.path.join(input_dir, "cam1", "data.csv"))
    imu_df = load_and_clean_csv(os.path.join(input_dir, "imu0", "data.csv"))
    ### NOW UNUSED ### --> using VIO output rather than ground-truth data
    # gt_df = load_and_clean_csv(
    #     os.path.join(input_dir, "state_groundtruth_estimate0", "data.csv")
    # )

    # Merge cam0 and cam1 by outer join
    combined = pd.merge(
        cam0_df, cam1_df, on="timestamp", how="outer", suffixes=("_cam0", "_cam1")
    )
    combined.sort_values("timestamp", inplace=True)
    combined["filename_cam0"] = combined["filename_cam0"].ffill()
    combined["filename_cam1"] = combined["filename_cam1"].ffill()

    # Interpolate IMU and GT to match combined camera timestamps
    ref_timestamps = combined["timestamp"]

    imu_interp = (
        imu_df.set_index("timestamp")
        .reindex(ref_timestamps, method="nearest")
        .reset_index()
    )
    ### NOW UNUSED ### --> using VIO output rather than ground-truth data
    # gt_interp = (
    #     gt_df.set_index("timestamp")
    #     .reindex(ref_timestamps, method="nearest")
    #     .reset_index()
    # )

    ### VIO OUTPUT ###
    # Load and process VIO output data
    vio_cols = ["timestamp_s", "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z"]
    vio_df = pd.read_csv(vio_csv_path, sep=' ', names=vio_cols)
    
    # Convert timestamp from seconds to nanoseconds with high precision
    # Using string manipulation to avoid floating point precision loss
    vio_df["timestamp"] = vio_df["timestamp_s"].apply(
        lambda x: int(str(x).replace('.', '').ljust(19, '0')[:19])
    )
    vio_df = vio_df.drop(columns=["timestamp_s"])
    
    # CRITICAL FIX: Reorder quaternion to match expected output format [q_x, q_y, q_z, q_w]
    vio_df = vio_df[["timestamp", "p_x", "p_y", "p_z", "q_x", "q_y", "q_z", "q_w"]]
    vio_df = vio_df.sort_values("timestamp")
    
    # Interpolate VIO output to match combined camera timestamps
    vio_interp = (
        vio_df.set_index("timestamp")
        .reindex(ref_timestamps, method="nearest")
        .reset_index()
    )

    # Merge interpolated IMU/VIO back into combined frame table
    combined = combined.merge(
        imu_interp, on="timestamp", how="left", suffixes=("", "_imu")
    )
    combined = combined.merge(
        vio_interp, on="timestamp", how="left", suffixes=("", "_gt")    # formerly gt_interp
    )

    # Optional: drop rows where VIO data is still missing (e.g. beginning or end of run)
    combined.dropna(subset=["p_x", "q_x"], inplace=True)

    # Final structure
    output = combined[
        [
            "timestamp",
            "filename_cam0",
            "filename_cam1",
            "p_x",
            "p_y",
            "p_z",
            "q_x",
            "q_y",
            "q_z",
            "q_w",
            "w_x",
            "w_y",
            "w_z",
            "a_x",
            "a_y",
            "a_z",
        ]
    ]
    output.to_csv(output_filepath, index=False)

def build_sequence(
    aStartPos: int, aEndPos: int, dataset: IMUImageDataset
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    data = [dataset[i] for i in range(aStartPos, aEndPos)]
    inputs, labels = zip(*data)
    imu_data, cam0_images, cam1_images = zip(*inputs)
    imu_data = torch.stack(imu_data)
    cam0_images = torch.stack(cam0_images)
    cam1_images = torch.stack(cam1_images)
    labels = torch.stack(labels)
    return [imu_data, cam0_images, cam1_images], labels

def build_all_random_batches(
    aStartPos: int,
    aEndPos: int,
    sequenceLength: int,
    batch_size: int,
    dataset,
    drop_last: bool = True,
    seed: int = None,
):
    """
    Builds randomly shuffled batches of sequences from a sequential dataset.

    Each sequence is of length `sequenceLength`, and each batch consists of `batch_size` such sequences.

    Parameters:
    - aStartPos: starting index in the dataset
    - aEndPos: ending index (exclusive)
    - sequenceLength: number of samples per sequence
    - batch_size: number of sequences per batch
    - dataset: an IMUImageDataset
    - drop_last: whether to drop the final batch if it's smaller than batch_size
    - seed: optional random seed for reproducibility
    """
    # 1. Extract all valid sequence start indices
    valid_starts = [
        i for i in range(aStartPos, aEndPos - sequenceLength + 1)
    ]

    if seed is not None:
        random.seed(seed)

    # 2. Shuffle them
    random.shuffle(valid_starts)

    # 3. Group into batches
    all_batches = []
    for i in range(0, len(valid_starts), batch_size):
        batch_indices = valid_starts[i : i + batch_size]
        if len(batch_indices) < batch_size and drop_last:
            break  # drop incomplete batch

        batch = [build_sequence(start, start + sequenceLength, dataset)
                 for start in batch_indices]

        # Some sequences might be None (invalid/missing); filter them
        batch = [b for b in batch if b is not None]
        if len(batch) < batch_size:
            continue  # skip batch with missing sequences

        imu_data, cam0_images, cam1_images = zip(*[b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        imu_data = torch.stack(imu_data)
        cam0_images = torch.stack(cam0_images)
        cam1_images = torch.stack(cam1_images)
        all_batches.append(([imu_data, cam0_images, cam1_images], labels))

    return all_batches

class FlightDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )

