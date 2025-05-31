import pandas as pd
import os
import requests
import zipfile
import random
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class IMUImageDataset(Dataset):
    def __init__(self, csv_path, cam0_image_root, cam1_image_root, transform=None):
        self.data = pd.read_csv(csv_path)
        self.cam0_image_root = cam0_image_root
        self.cam1_image_root = cam1_image_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        cam0_path = os.path.join(self.cam0_image_root, row['filename_cam0'])
        cam0_image = Image.open(cam0_path)
        cam0_image = self.transform(cam0_image)

        cam1_path = os.path.join(self.cam1_image_root, row['filename_cam1'])
        cam1_image = Image.open(cam1_path)
        cam1_image = self.transform(cam1_image)

        # Load IMU features
        imu = row[["w_x", "w_y", "w_z", "a_x", "a_y", "a_z"]].values.astype("float32")
        imu_tensor = torch.tensor(imu)

        ### no longer using ###
        # Load ground truth features
        # ground_truth = row[["p_x", "p_y", "p_z", "q_x", "q_y", "q_z", "q_w"]].values.astype("float32")
        # ground_truth_tensor = torch.tensor(ground_truth)

        ### VIO OUTPUT ###
        # Load pose label (from VIO output)
        pose_label = row[["p_x", "p_y", "p_z", "q_x", "q_y", "q_z", "q_w"]].values.astype("float32")
        pose_label_tensor = torch.tensor(pose_label)

        return [imu_tensor, cam0_image, cam1_image], pose_label_tensor

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

import random
import torch

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

if __name__ == "__main__":
    # Example usage
    # vicon_room_1_easy_url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip"
    # vicon_room_1_medium_url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip"
    # vicon_room_1_difficult_url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip"
    # download_dataset(
    #     "vicon_room_1_easy",
    #     vicon_room_1_easy_url
    # )
    # download_dataset(
    #     "vicon_room_1_medium",
    #     vicon_room_1_medium_url
    # )
    # download_dataset(
    #     "vicon_room_1_difficult",
    #     vicon_room_1_difficult_url
    # )


    ### UPDATED WITH VIO OUTPUT ###
    # Prepare the combined CSV files for each dataset
    # Check if the CSV files already exist to avoid reprocessing

    # prep_combined_csv(
    #     os.path.join("..", "data", "vicon_room_1_easy", "mav0"),
    #     os.path.join("..", "data", "vins_output", "V1_01_easy.csv"),          # new
    #     os.path.join("..", "data", "vicon_room_1_easy", "combined.csv")
    # )
    # prep_combined_csv(
    #     os.path.join("..", "data", "vicon_room_1_medium", "mav0"),
    #     os.path.join("..", "data", "vins_output", "V1_02_medium.csv"),        # new
    #     os.path.join("..", "data", "vicon_room_1_medium", "combined.csv")
    # )
    # prep_combined_csv(
    #     os.path.join("..", "data", "vicon_room_1_difficult", "mav0"),
    #     os.path.join("..", "data", "vins_output", "V1_03_difficult.csv"),     # new
    #     os.path.join("..", "data", "vicon_room_1_difficult", "combined.csv")
    # )

    # Load the datasets
    easy_dataset = IMUImageDataset(
        csv_path=os.path.join("..", "data", "vicon_room_1_easy", "combined.csv"),
        cam0_image_root=os.path.join("..", "data", "vicon_room_1_easy", "mav0", "cam0", "data"),
        cam1_image_root=os.path.join("..", "data", "vicon_room_1_easy", "mav0", "cam1", "data")
    )
    print(f"Easy dataset of size: {len(easy_dataset)} loaded successfully.")

    medium_dataset = IMUImageDataset(
        csv_path=os.path.join("..", "data", "vicon_room_1_medium", "combined.csv"),
        cam0_image_root=os.path.join("..", "data", "vicon_room_1_medium", "mav0", "cam0", "data"),
        cam1_image_root=os.path.join("..", "data", "vicon_room_1_medium", "mav0", "cam1", "data")
    )
    print(f"Medium dataset of size: {len(medium_dataset)} loaded successfully.")

    difficult_dataset = IMUImageDataset(
        csv_path=os.path.join("..", "data", "vicon_room_1_difficult", "combined.csv"),
        cam0_image_root=os.path.join("..", "data", "vicon_room_1_difficult", "mav0", "cam0", "data"),
        cam1_image_root=os.path.join("..", "data", "vicon_room_1_difficult", "mav0", "cam1", "data")
    )
    print(f"Difficult dataset of size: {len(difficult_dataset)} loaded successfully.")

    # Example of building a batch
    easy_batches = build_all_random_batches(
        aStartPos=0,
        aEndPos=100, # TODO: Change this to len(easy_dataset) for full dataset
        sequenceLength=10,
        batch_size=32,
        dataset=easy_dataset,
        drop_last=True,
        seed=42
    )
    print(f"Number of easy batches: {len(easy_batches)}")

    medium_batches = build_all_random_batches(
        aStartPos=0,
        aEndPos=100,# TODO: Change this to len(medium_dataset) for full dataset       sequenceLength=10,
        sequenceLength=10,
        batch_size=32,
        dataset=medium_dataset,
        drop_last=True,
        seed=42
    )
    print(f"Number of medium batches: {len(medium_batches)}")
    
    difficult_batches = build_all_random_batches(
        aStartPos=0,
        aEndPos=100, # TODO: Change this to len(difficult_dataset) for full dataset
        sequenceLength=10,
        batch_size=32,
        dataset=difficult_dataset,
        drop_last=True,
        seed=42
    )
    print(f"Number of difficult batches: {len(difficult_batches)}")

    # Split the datasets into training and testing sets
    easy_train_batches, easy_test_batches = train_test_split(
        easy_batches, test_size=0.2, shuffle=True, random_state=42
    )
    print(f"Number of easy train batches: {len(easy_train_batches)}")
    print(f"Number of easy test batches: {len(easy_test_batches)}")

    medium_train_batches, medium_test_batches = train_test_split(
        medium_batches, test_size=0.2, shuffle=True, random_state=42
    )
    print(f"Number of medium train batches: {len(medium_train_batches)}")
    print(f"Number of medium test batches: {len(medium_test_batches)}")

    difficult_train_batches, difficult_test_batches = train_test_split(
        difficult_batches, test_size=0.2, shuffle=True, random_state=42
    )
    print(f"Number of difficult train batches: {len(difficult_train_batches)}")
    print(f"Number of difficult test batches: {len(difficult_test_batches)}")

    # Example of accessing a batch
    first_easy_train_batch = easy_train_batches[0]
    first_easy_train_inputs, first_easy_train_labels = first_easy_train_batch
    imu_data = first_easy_train_inputs[0]  # Shape: [batch_size, seq_len, M]
    camera0_images = first_easy_train_inputs[1]  # Shape: [batch_size, seq_len, C, H, W]
    camera1_images = first_easy_train_inputs[2]  # Shape: [batch_size, seq_len, C, H, W]
    print(f"IMU Data Shape: {imu_data.shape}")
    print(f"Camera 0 Images Shape: {camera0_images.shape}")
    print(f"Camera 1 Images Shape: {camera1_images.shape}")


