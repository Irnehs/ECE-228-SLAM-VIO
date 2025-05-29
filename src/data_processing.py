import pandas as pd

def load_euroc_imu(filepath):
    """
    Load IMU data from EuRoC .csv file (imu0/data.csv).
    Returns a DataFrame with columns: ax, ay, az, gx, gy, gz
    """
    df = pd.read_csv(filepath, comment='#', header=None)
    df.columns = ['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']
    return df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']]

def normalize_imu(df):
    """
    Normalize IMU columns: ax, ay, az, gx, gy, gz
    """
    return (df - df.mean()) / df.std()
