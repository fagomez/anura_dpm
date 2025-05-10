<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <header>
    <h1>&#x1F438; Diffusion-based data augmentation for anuran call generation</h1>
  </header>


# README

## Repository Workflow

This repository contains code organized in a series of numbered folders. Follow the steps below to run the workflow in numerical order in the Directory code:
1. **Folder 1: Data Preprocessing**
    - Unzip the data using `code/1_preprocessing/0_unzip.py`.
    - To understand the raw data, use the Jupyter notebook `code/1_preprocessing/1_humboldt_data_understanding.ipynb`.
    - To build the training data for each class (BOAFAB and other frog calls), use the following Jupyter notebooks:
        - BOAFAB: `code/1_preprocessing/2_data_selection_final_to_train_BOAFAB.ipynb`. Creates the folder `data/train_gen/BOAFAB/*`.
        - OTHER: `code/1_preprocessing/2_data_selection_final_to_train_OTHER.ipynb`. Creates the folder `data/train_gen/OTHER/*`.
    - To split the training data into training, test, and validation sets, use the Jupyter notebook `code/1_preprocessing/4_data_split_to_CLASSIFIER.ipynb`. Creates the folder `data/classifier/*`.

2. **Folder 2: Train Generative Model**
    - To train the Diffusion Model, run the Jupyter notebook `code/2_generative/TRAIN_A_DIFF_BOAFAB.ipynb`

3. **Folder 3: Train Classification Model (for evaluation)**
    - To train the classification model, run the Jupyter notebook `code/3_classifier/1_run_model_no_data_aug.ipynb`.
    - To evaluate the classification model, run the Jupyter notebook `code/3_classifier/2_eval_model.ipynb`.

4. **Folder 4: Systematic Selection**
    - To generate a sample of generated audios and take a sample of real audios, run the Jupyter notebook `code/4_systematic_selection/1_generate_samples.ipynb`. Creates the floder `/data/embedding/BOAFAB_GENERATED/`, and `data/embedding/BOAFAB_REAL/`.
    - To run the systematic selection algorithm, run the Jupyter notebook `code/4_systematic_selection/2_final_figures_embedding_classifier_nodataaug.ipynb`
  
5. **Folder 5: Classification under data scarcity**
    - To create different diffusion models on the different species `code/5_classification_under_data_scarcity/1_exp_augmentation.ipynb`.
    - To generate data using diffusion models `code/5_classification_under_data_scarcity/2_gen_data_diffusion.ipynb`.
    - To run classification experiments of data scarcity `code/5_classification_under_data_scarcity/3_classification_augmentation.ipynb`.

6. **Folder 5: CNN-WGANS**
    - Code for running the CNN-WGANS.
