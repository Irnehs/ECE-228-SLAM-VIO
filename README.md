# Task 2: SLAM VIO Path Error Optimization

## Authors (Group 16)
- Matthew Alegrado  
- Daniel Sanei  
- Henri Schulz  
- Veeral Patel  

## Repo Structure
```
.
├── catkin_ws
│   └── src
│       ├── CMakeLists.txt -> /opt/ros/noetic/share/catkin/cmake/toplevel.cmake
│       └── VINS-Fusion
├── config.yaml
├── data
│   ├── ground_truth
│   │   ├── V1_01_easy_vio_vs_gt_error.csv
│   │   ├── V1_02_medium_vio_vs_gt_error.csv
│   │   └── V1_03_difficult_vio_vs_gt_error.csv
│   ├── vicon_room_1_difficult
│   │   ├── combined.csv
│   │   └── mav0
│   ├── vicon_room_1_easy
│   │   ├── combined.csv
│   │   └── mav0
│   ├── vicon_room_1_medium
│   │   ├── __MACOSX
│   │   ├── combined.csv
│   │   └── mav0
│   └── vins_output
│       ├── V1_01_easy.csv
│       ├── V1_02_medium.csv
│       ├── V1_03_difficult.csv
│       └── vio.csv
├── download_all_data.py
├── environment.yml
├── final_gt.ckpt
├── final_model_gt.pth
├── final_model_vio.pth
├── final_vio.ckpt
├── logs
│   └── lightning_logs
├── model_testing.ipynb
├── models
│   ├── Decoder.py
│   ├── FusionRNN.py
│   ├── ImageEncoder.py
│   ├── IMUEncoder.py
│   └── SLAMErrorPredictor.py
├── notebooks
│   ├── data_pipeline.ipynb
│   ├── model_architecture_diagram.png
│   ├── model_architecture.ipynb
│   └── model-architecture-mermaid.md
├── output_gt.csv
├── plot.py
├── README.md
├── run.py
├── src
│   └── squared_error.py
├── testing_output.csv
├── tools
├── train loss.csv
├── val loss.csv
└── validation_output.csv
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
