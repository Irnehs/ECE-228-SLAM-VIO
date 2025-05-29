import torch
from src.data_processing import load_euroc_imu, normalize_imu

def prepare_sequence(filepath, sequence_length=50):
    df = load_euroc_imu(filepath)
    normed = normalize_imu(df)
    data = normed.to_numpy()
    num_sequences = data.shape[0] - sequence_length + 1
    sequences = [data[i:i+sequence_length] for i in range(num_sequences)]
    batch = torch.tensor(sequences, dtype=torch.float32)
    return batch
# 
#10x6, 
#6x10 is the input shape for the IMUEncoder
def prepare_batch(filepath, batch_size=32, sequence_length=50):
    batch = prepare_sequence(filepath, sequence_length)
    num_batches = batch.shape[0] // batch_size
    batches = batch[:num_batches * batch_size].view(num_batches, batch_size, sequence_length, -1)
    return batches