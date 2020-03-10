CSE 517 - Natural Language Processing Project
Winter 2020

# Reproducing "Language Models as Knowledge Bases"
The original code base can be found [here](https://github.com/facebookresearch/LAMA), and referenced paper [here](https://www.aclweb.org/anthology/D19-1250.pdf).  

## To get started:
### 1. Download data
```bash
wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
rm data.zip
```

### 2. Download models
Install spacy model
```bash
python3 -m spacy download en
```

To download a given model:
```bash
cd model_download
chmod +x <insert_script_for_model>
./<insert_script_for_model>
```
The script will create and populate a <code>pretrained\_lms</code> directory with the chosen model to download.

To avoid exceeding disk space, download only one model at a time.  When finished working with a model, run 
```bash
./clean_up_models.sh
```
to clear out any previously-downloaded models. 

To build up the common vocabulary (intersection of all models' vocabularies), run ... TODO HERE! 
