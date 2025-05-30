from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
import pandas as pd
import requests
import zipfile
from tqdm import tqdm
import numpy as np


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
        cam0_image = Image.open(cam0_path).convert("RGB")
        cam0_image = self.transform(cam0_image)

        cam1_path = os.path.join(self.cam1_image_root, row['filename_cam1'])
        cam1_image = Image.open(cam1_path).convert("RGB")
        cam1_image = self.transform(cam1_image)

        # Load IMU features
        imu = row[["w_x", "w_y", "w_z", "a_x", "a_y", "a_z"]].values.astype("float32")
        imu_tensor = torch.tensor(imu)

        # Load ground truth features
        ground_truth = row[["p_x", "p_y", "p_z", "q_x", "q_y", "q_z", "q_w"]].values.astype("float32")
        ground_truth_tensor = torch.tensor(ground_truth)

        return [imu_tensor, cam0_image, cam1_image], ground_truth_tensor


def prep_combined_csv(input_dir, output_filepath):
    def load_and_clean_csv(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df.rename(columns={
            '#timestamp [ns]': 'timestamp',
            '#timestamp': 'timestamp',
            'p_RS_R_x [m]': 'p_x',
            'p_RS_R_y [m]': 'p_y',
            'p_RS_R_z [m]': 'p_z',
            'q_RS_x []': 'q_x',
            'q_RS_y []': 'q_y',
            'q_RS_z []': 'q_z',
            'q_RS_w []': 'q_w',
            'w_RS_S_x [rad s^-1]': 'w_x',
            'w_RS_S_y [rad s^-1]': 'w_y',
            'w_RS_S_z [rad s^-1]': 'w_z',
            'a_RS_S_x [m s^-2]': 'a_x',
            'a_RS_S_y [m s^-2]': 'a_y',
            'a_RS_S_z [m s^-2]': 'a_z',
        }, inplace=True)
        df['timestamp'] = df['timestamp'].astype(np.int64)
        df.sort_values('timestamp', inplace=True)
        return df

    # Load and clean
    cam0_df = load_and_clean_csv(os.path.join(input_dir, 'cam0', 'data.csv'))
    cam1_df = load_and_clean_csv(os.path.join(input_dir, 'cam1', 'data.csv'))
    imu_df  = load_and_clean_csv(os.path.join(input_dir, 'imu0', 'data.csv'))
    gt_df   = load_and_clean_csv(os.path.join(input_dir, 'state_groundtruth_estimate0', 'data.csv'))

    # Merge cam0 and cam1 by outer join
    combined = pd.merge(cam0_df, cam1_df, on='timestamp', how='outer', suffixes=('_cam0', '_cam1'))
    combined.sort_values('timestamp', inplace=True)
    combined['filename_cam0'] = combined['filename_cam0'].ffill()
    combined['filename_cam1'] = combined['filename_cam1'].ffill()

    # Interpolate IMU and GT to match combined camera timestamps
    ref_timestamps = combined['timestamp']

    imu_interp = imu_df.set_index('timestamp').reindex(ref_timestamps, method='nearest').reset_index()
    gt_interp = gt_df.set_index('timestamp').reindex(ref_timestamps, method='nearest').reset_index()

    # Merge interpolated IMU/GT back into combined frame table
    combined = combined.merge(imu_interp, on='timestamp', how='left', suffixes=('', '_imu'))
    combined = combined.merge(gt_interp, on='timestamp', how='left', suffixes=('', '_gt'))

    # Optional: drop rows where GT data is still missing (e.g. beginning or end of run)
    combined.dropna(subset=['p_x', 'q_x'], inplace=True)

    # Final structure
    output = combined[['timestamp', 'filename_cam0', 'filename_cam1', 'p_x', 'p_y', 'p_z', 'q_x', 'q_y', 'q_z', 'q_w', 'w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z']] 
    output.to_csv(output_filepath, index=False)
    

def download_dataset(name, url):
    data_dir = os.path.join("..", "data", name)
    zip_path = os.path.join(data_dir, f"{name}.zip")

    print(f"Setting up dataset: {name}")
    os.makedirs(data_dir, exist_ok=True)

    # Streamed download with progress bar
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, "wb") as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=name, ncols=80
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    # Unzip into the subdirectory
    print(f"Unzipping into {data_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Remove the zip file
    print("Removing archive...")
    os.remove(zip_path)

    print(f"✅ Dataset '{name}' ready in {data_dir}")


def build_sequence(aStartPos : int, aEndPos : int, dataset : IMUImageDataset) -> dict[str, torch.Tensor]:
    data = [dataset[i] for i in range(aStartPos, aEndPos)]
    inputs, labels = zip(*data)
    imu_data, cam0_images, cam1_images = zip(*inputs)
    imu_data = torch.stack(imu_data)    
    cam0_images = torch.stack(cam0_images)
    cam1_images = torch.stack(cam1_images)
    labels = torch.stack(labels)
    return [imu_data, cam0_images, cam1_images], labels

# Example usage
# download_dataset(
#     "vicon_room_1",
#     "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip"
# )