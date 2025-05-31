# Imports
import pandas as pd
import numpy as np
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)

# List of datasets and corresponding file paths
datasets = [
    {
        "name": "V1_01_easy",
        "vio_csv": "../data/vins_output/V1_01_easy.csv",
        "gt_csv": "../data/ground_truth/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv",
        "result_csv": "../data/ground_truth/V1_01_easy_vio_vs_gt_error.csv"
    },
    {
        "name": "V1_02_medium",
        "vio_csv": "../data/vins_output/V1_02_medium.csv",
        "gt_csv": "../data/ground_truth/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv",
        "result_csv": "../data/ground_truth/V1_02_medium_vio_vs_gt_error.csv"
    },
    {
        "name": "V1_03_difficult",
        "vio_csv": "../data/vins_output/V1_03_difficult.csv",
        "gt_csv": "../data/ground_truth/V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv",
        "result_csv": "../data/ground_truth/V1_03_difficult_vio_vs_gt_error.csv"
    }
]

# Function for determining orientation error for each row
def quat_angle_error(row):
    # Get quaternion components
    q1 = np.array([row['qw_vio'], row['qx_vio'], row['qy_vio'], row['qz_vio']])
    q2 = np.array([row['qw_gt'], row['qx_gt'], row['qy_gt'], row['qz_gt']])
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    # Compute absolute dot product
    dot_product = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    # Compute angle (radians)
    angle = 2 * np.arccos(dot_product)
    return angle

# Function for processing VIO, ground-truth (GT) datasets and matching timestamps
        # Compute squared error between VIO and GT
def squared_error(vio_csv, gt_csv, result_csv):

    # Load VIO system output
    vio_cols = ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
    vio_df = pd.read_csv(vio_csv, sep=' ', names=vio_cols)
    vio_df['timestamp'] = vio_df['timestamp'].astype(float)     # convert timestamp to float

    # Load ground-truth data
    gt_df = pd.read_csv(gt_csv, comment='#')
    gt_df = gt_df.rename(columns={              # Rename ground truth columns
        gt_df.columns[0]: 'timestamp',          # only need first 8, the rest are velocity, biases, etc.
        gt_df.columns[1]: 'x',
        gt_df.columns[2]: 'y',
        gt_df.columns[3]: 'z',
        gt_df.columns[4]: 'qw',
        gt_df.columns[5]: 'qx',
        gt_df.columns[6]: 'qy',
        gt_df.columns[7]: 'qz'
    })
    gt_df['timestamp'] = gt_df['timestamp'].astype(float) / 1e9     # Convert ground-truth from nanoseconds to seconds

    # Sort datasets by timestamps
    vio_df = vio_df.sort_values('timestamp')
    gt_df = gt_df.sort_values('timestamp')

    # Merge two datasets on nearest timestamp
    merged = pd.merge_asof(
        vio_df.sort_values('timestamp'),
        gt_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        suffixes=('_vio', '_gt'),
        tolerance=0.000001 # 100 us tolerance
    )
    merged = merged.dropna()    # Drop rows with no match

    # Compute position error (meters squared)
    merged['pos_err'] = (
        (merged['x_vio'] - merged['x_gt'])**2 +
        (merged['y_vio'] - merged['y_gt'])**2 +
        (merged['z_vio'] - merged['z_gt'])**2
    )   # Euclidean distance

    # Compute orientation error (angular difference, radians)
    merged['orient_err'] = merged.apply(quat_angle_error, axis=1)

    # Save result
    merged[['timestamp', 'pos_err', 'orient_err']].to_csv(result_csv, index=False)

    # Display results
    print(f"\nProcessed {result_csv}")
    print("VIO rows:", len(vio_df))
    print("GT rows:", len(gt_df))
    print("Merged rows:", len(merged))
    print(merged[['timestamp', 'pos_err', 'orient_err']].head())

# Run for all datasets (easy, medium, hard)
for ds in datasets:
    squared_error(ds['vio_csv'], ds['gt_csv'], ds['result_csv'])