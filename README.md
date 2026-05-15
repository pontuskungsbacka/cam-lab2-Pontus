# Machine Learning Assigment 2: CNN
This repository contains my completed Assignment 2 focusing on interpreting Convolutional Neural Networks (CNNs) using Class Attribution Maps.

## Analysis & Purpose
In my report <b>[report.ipynb](/report.ipynb)</b> I analyses three ImageNet classes: 
  - lesser_panda Index: 387
  - spaghetti_squash Index: 940
  - warplane Index: 895



The purpose of the report is to see if the CNN can really predict correctly and what the different layers is.

## Contents
- `report.ipynb` - The Report contains the analysis with discussion, visualisations, results, and conclusions. 
- `processing.py` - 
- `src/imagenet_class_index.json` 
- `data/` 
- `README.md` 


## Requirements
To run the files, you need:

- ``python 3.11.14``
- ``torch``
- ``torchvision``
- ``torchcam``
- ``pillow``
- ``matplotlib``

### Recommendation
 If you want to clone this repository and look at the Jupyter Notebook and test the python script,  use this step.<br/>
<small>Command is for Windows Users</small><br/> 
- In command line in terminal write: <br />``git clone https://github.com/pontuskungsbacka/cam-lab2-Pontus.git``
- Go to the folder: ``cd cam-lab2-Pontus``
- Install ``uv venv .venv`` in terminal for the folder.
- Activate the venv: <br/> ``.venv\Scripts\activate`` 

- Install the correct python version: <br/> ``uv python 3.11.14``
- Install libaries need from requirements.txt: <br/> ``uv pip install -r requirements.txt``
- Open repo in VScode: <br/> ``code .``