# Task 2: SLAM VIO Path Error Optimization

## Authors (Group 16)
- Matthew Alegrado  
- Daniel Sanei  
- Henri Schulz  
- Veeral Patel  

## Repo Structure
```
ECE-228-SLAM-VIO-2/
├── data/
│   └── data.csv
├── models/
│   ├── Decoder.py
│   ├── FusionRNN.py
│   ├── ImageEncoder.py
│   ├── IMUEncoder.py
│   └── SLAMErrorPredictor.py
├── notebooks/
│   ├── data_pipeline.ipynb
│   ├── model_testing.ipynb
│   ├── data_pipeline.pdf
│   ├── model_testing.pdf
│   └── model_architecture_diagram.png
├── src/
│   ├── __pycache__/
├── tools/
├── config.yaml
├── environment.yml
├── download_all_data.py
├── run.py
├── README.md
└── .gitignore
```

Note that `data/` must be locally populated and are ignored by Git to prevent large file uploads.
## Dependency Management 

We are using `conda` to manage out dependencies for this project.

To recreate the environment locally, run
```bash
conda env create -f environment.yml
conda activate ece228-slam-vio 
```

When installing packages, make sure to install using 
```bash
conda install <package-name>
``` 
and then rebuild the `environment.yml` file using 
```bash
conda env export --from-history > environment.yml
```

## Sample DataFrame Format (`data/data.csv`)

| Timestamp | ACC_X | ACC_Y | ACC_Z | GYRO_X | GYRO_Y | GYRO_Z | Img_L | Img_R | Pos_X | Pos_Y | Pos_Z | Q_w | Q_x | Q_y | Q_z |
|-----------|-------|-------|-------|--------|--------|--------|--------|--------|--------|--------|--------|-----|-----|-----|-----|
| 1403636.2 | ...   | ...   | ...   | ...    | ...    | ...    | path/to/img1_L.png | path/to/img1_R.png | 0.1 | 0.2 | 0.3 | 1.0 | 0.0 | 0.0 | 0.0 |

## Model Overview
- **IMU Encoder**: Bidirectional LSTM
- **Image Encoders**: MobileNetV2 backbones
- **Fusion RNN**: Temporal fusion of modalities
- **Decoder**: RNN decoder predicting K=10 pose steps

## Future Work
- Add real-time inference and ROS integration
- Extend to additional flight scenes
- Incorporate contrastive loss
