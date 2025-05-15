# SLAM VIO Path Error Optimization

## File Structure

```
.
├── data
│   └── somedata.txt
├── environment.yml
├── models
│   └── sometrainedmodel.dat
├── notebooks
│   └── data_pipeline.ipynb
├── README.md
└── src
    └── data_processing.py
```
Note that `data/` and `models/` must be locally populated and are ignored by Git to prevent large file uploads.
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