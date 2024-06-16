# RNAfold 

## Introduction
This project focuses on training and inference processes for predicting RNA structures using Variational Autoencoder. It includes description of configurations, training, inference, and output files.


## Getting started 

Clone repository 

```bash
git clone 
```

## Usage

1. **Training**: 

Use the configuration file `config.json` and run `pipeline.py` for training. The device is automatically determined. If GPU is available, the GPU is used for training. Otherwise, CPU is used.

```bash 
python src/pipeline.py
```

2. **Inference**: 

Use the configuration file `config_inference.json` and run `inference.py` for inference.


```bash 
python src/inference.py
```

## Configuration
The configuration file `config.json` is essential for both training and inference processes. It includes parameters such as batch size, number of epochs, data specifications, model architecture details, paths, and more.

## Configuration Parameters
- **Description**: Description of the experiment.
- **batch_size**: Batch size for the experiment.
- **n_epoch**: Number of epochs for the experiment.
- **max_data**: Maximum number of sequences in the experiment.
- **min_length**: Minimal length of the sequence.
- **max_length**: Maximal length of the sequence.
- **n_z**: Hidden dimension of the architecture.
- **SEED**: Random seed for reproducibility.
- **beta**: Parameter for the loss function.
- **dropout**: Dropout rate.
- **ratio_arr**: Cumulative proportion of training/validation/test dataset.
- **pdb_folder_path**: Folder where the pdb files are stored.
- **exp_path**: Experiment name.
- **logs**: Log file.
- **proportion**: Proportion of the dataset used for training (for debugging purposes).
- **file_generated**: Name of the file where the examples are stored.
- **n_images_show**: Number of images to show in `file_generated`.
- **jobs**: Parallel jobs for CPU training.




## Output Files
The same output is produced for the training and inference pipeline. The structure is the following:


<br/>
├─ config/ <br/>
│   ├─ config.json: Experiment configuration settings.<br/>
│   └─ vae.py: Variational Autoencoder (VAE) implementation for the experiment.<br/>
│<br/>
├── images/<br/>
│   ├── Predicted images for the test dataset, each scaled by a factor of 100.<br/>
│   └── images_comparison/<br/>
│       ├── Visual representations of distance matrices for test dataset sequences.<br/>
│       └── Different noise vector variations of the distance matrices.<br/>
│<br/>
├── model/<br/>
│   └── best_model.pth: The trained model file.<br/>
│<br/>
├── results/<br/>
│   ├── examples.jpg: Side-by-side comparison of predicted images against true images.<br/>
│   ├── logs.csv: Log file containing metrics like MAE, loss, and RMSE for both training and validation phases.<br/>
│   ├── loss.jpg: Graphical representation of the loss metric over the training and validation periods.<br/>
│   └── mae.jpg: Graphical representation of the Mean Absolute Error (MAE) over the training and validation periods.<br/>
│<br/>
└── test_calcs/<br/>
    ├── five.pickle: Generated distance matrix predictions for various test data points.<br/>
    ├── pred.pickle: Predicted distance matrices for the test dataset.<br/>
    ├── sequences.pickle: Sequence data for the test dataset.<br/>
    ├── test_results.txt: Performance metrics evaluated on the test dataset.<br/>
    └── true.pickle: Actual distance matrices for the test dataset.<br/>


## Conclusion
This README provides an overview of the RNAfold project, including its purpose, configuration, usage instructions, and output files structure.

For detailed information, refer to specific files and folders within the project.