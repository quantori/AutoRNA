# RNAfold 

## Introduction
This project is a pipeline for predicting tertiary RNA structure using a Variational Autoencoder.

## Getting started 

Clone repository 

```bash
git clone https://github.com/quantori/RNAfold.git
cd RNAfold
```

## Install dependencies
To install the required dependencies, use the requirements.txt file:

```bash
pip install -r requirements.txt
```
## Usage

**Training**: 

Use and change the configuration file `src/config/config.json` and run `src/pipeline.py` for training. The device is automatically determined. If GPU is available, the GPU is used for training. Otherwise, CPU is used.

```bash 
python src/pipeline.py
```

**Inference**: 

Use and change the configuration file `src/config/config_inference.json` and run `src/inference.py` for inference.


```bash 
python src/inference.py
```

Weights of models are available in link https://drive.google.com/file/d/1Si9N7BU08Ft_MyLmtfKD4dVLVIxgtmfI/view?usp=sharing

## Data

You can download the full dataset of (.ent files) using the following link: https://drive.google.com/file/d/1_uZ9coIGNIVSaW4-W0h4ZQYPE9Bm-Geu/view?usp=drive_link .<br/>
The size of the archived folder is 1.2GB, the size of unarchived folder is around 5GB. Please copy the folder **rna_data_v1** to **data** folder.
The folder **rna_data_v1_small** that is stored on GitHub is a small representative subset of the large dataset. The goal of this dataset is to show how the algorithm works. Please note, that running inference script on the rna_data_v1_small will lead to smaller error due to the data leakage (the dataset was formed as a part of the whole dataset before division into train/val/test).

## Configuration
The configuration file config.json is essential for both training and inference processes. It includes parameters such as batch size, number of epochs, data specifications, model architecture details, paths, and more. You use only the model_path, output_path,  pdb_path, and SEED for the inference pipeline. 

### Configuration Parameters for Training
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

### Configuration Parameters for Inference

- **batch_size**: Batch size for the experiment.
- **SEED**: Random seed for reproducibility.
- **pdb_folder_path**: Folder where the pdb files are stored.
- **output_path**: The folder for the results
- **model_path**: The path for the trained model



## Repository Structure
The same output is produced for the training and inference pipeline. The structure of the folder is the following:


 - **data** : the folders with pdb files(omitted for GitHub due to the size limits)<br/>
 - **experiments**  : the folder with the experiments<br/>
    - **some_experiment_name**<br/>
 
 
 - **src** : the source files
   - **config** : the config folder for training and inference pipelines)
   - **preprocessing**: the data preprocessing is happening here
   - **utils** : the utils files


## Output

The same output is produced for the training and inference pipeline. Here is the example of the folders produced:
- **AE_random_nz_1024_dropout0.5beta1.1_version_final**<br/>
     - **config** : saved config<br/>
     - **images** : produced images for test dataset (heatmaps)<br/>
     - **images_comparison**: produced images for test dataset for different noise vectors<br/>
     - **model** : trained model is stored in the folder <br/>
     - **results**: graphical results for the loss function (mae)<br/>
     - **test_calcs** : the saved pickle files for the visualiation<br/>

## Citation

Kazanskii, M. A., Uroshlev, L., Zatylkin, F., Pospelova, I., Kantidze, O., & Gankin, Y. (2024). RNAfold: RNA tertiary structure prediction using variational autoencoder. bioRxiv. https://doi.org/10.1101/2024.06.18.599511
## Contact Information
Feel free to contact us at [maxim.kazanskiy@quantori.com](mailto:maxim.kazanskiy@quantori.com).
