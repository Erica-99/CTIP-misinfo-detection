# CTIP-misinfo-detection

This is the repository for COS30049 Computing Technology Innovation Project Assignment 2.

Contributors:
- Erica Thompson
- Dinuka Gunarathne
- Sahan Maduranga Mayadunna

The below datasets were used to create the processed one provided in this repository. Following the instructions below, the processed dataset can be recreated. Observe `loading_datasets.ipynb` for more information.

1. The provided dataset
    - Place the three excel files in `data/provided/`
2. https://www.kaggle.com/datasets/vishakhdapat/fake-news-detection
    - Place at `data/fake-news-detection_10k/fake_and_real_news.csv`
3. https://www.kaggle.com/datasets/techykajal/fakereal-news
    - Place at `data/fake-real-news_10k/dataset.csv`
4. https://www.kaggle.com/datasets/stevenpeutz/misinformation-fake-news-text-dataset-79k
    - Place the two files at `data/misinformation-fake-news-text-dataset_79k/`
    - `DataSet_Misinfo_FAKE.csv`
    - `DataSet_Misinfo_TRUE.csv`

## Python Environment Setup

### 1. Set up a new virtual environment with conda
Run the below command in a powershell/shell/terminal window with anaconda installed.

The environment can be called whatever you want, this is just a suggestion. Please use Python version `3.12.0` for maximum compatibility.
```
conda create -n ctip-s20-g4 python==3.12.0
```

### 2. Activate the environment
Use the environment name you set in step 1.
```
conda activate ctip-s20-g4
```

### 3. Install requirements
Make sure you navigate your command prompt window to this folder before installing requirements.
```
cd CTIP-misinfo-detection
```
Now install requirements:
```
pip install -r requirements.txt
```
Note it may take a while to install these, especially if you do not have many of the libraries cached. Jupyter in particular has a very large amount of dependencies and can take a while to install.

## Data Processing
All data processing is done in the `loading_datasets.ipynb` Jupyter notebook. Halfway down the notebook there is a cell which loads a csv dataset into memory right before the cell where all the processing happens. If you want to perform additional processing on the dataset, direct the loading function to your dataset's filepath and then add whatever processing functions you like to the following cell. 

## Data Visualisation
Data visualisation happens in the `data_visualisation.ipynb` Jupyter notebook. After importing the necessary libraries at the top and reading in the dataset in the following cell, each cell should be able to be run independently to view further visualisations.

## Model Training
Most model training is conducted in the `model_training.ipynb` Jupyter notebook. After running the import and config sections near the top of the notebook, most sections can be run independently, though within sections cells often rely on the previous. The only exceptions to this are the Hybrid pipeline which depends on the features extracted in the manual feature extraction section, and the clustering section, which relies on the training of a semantic embedding model. 

Additionally, some training and evaluation of classifier models can be found in the `pross_train.py` file, thought this was largely refactored into the main notebook. The same is true for `kmeans_word2vec.py`.

If you would like to run training of our final selected classification model, please run the `train_classifier.py` file. It will automatically train, save, and show evaluation results for our Linear SVM model with TF-IDF feature extraction.

## Model Inference
Inference with our Linear SVM classifier model can be done using the `inference_classifier.py` script. It will use the saved model in the folder.

This script can take several command-line arguments:

- `--use-dataset`: This is a boolean flag which will tell the script to load a random post from the saved dataset and run inference on that, providing you with both the text and the result.
- `--use-text "Your text here"`: This parameter can be used to directly input text to be inferenced on. Please note that the `--use-dataset` flag will override this one. 

If neither flag is set, the default behaviour is to use the saved example text in the script, which is an excerpt from one of the posts in our dataset. Feel free to change this text as well by editing the `INPUT_TEXT` variable near the top of the script.