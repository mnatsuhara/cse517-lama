CSE 517 - Natural Language Processing Project

Winter 2020

Miya Natsuhara (mnats), Sophie Tian (shuxut)

# Reproducing "Language Models as Knowledge Bases"
The original code base can be found [here](https://github.com/facebookresearch/LAMA), and referenced paper [here](https://www.aclweb.org/anthology/D19-1250.pdf).  

## To get started:
### 1. Download the data
```bash
wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
rm data.zip
```

### 2. Download the models
Install spacy model
```bash
python3 -m spacy download en
```

Download the models to be compared:
```bash
chmod +x <insert_script_for_model>
./download_models.sh
```
The script will create and populate a <code>pre-trained\_language\_models</code> directory.  This script also uses the vocabulary files from each of the models downloaded to calculate the intersection of the vocabulary, and stores it in the same directory: <code>common\_vocab\_cased.txt</code> and <code>common\_vocab\_lowercased.txt</code> for the respectively cased models.

### 3. Run the models
To run a model on the different datasets, use
```bash
python3 scripts/run_experiments.py <model>
```
where you can specify the model you want to run by replacing <code><model></code> with one of <code>transformerxl, gpt2-xl, elmo, elmo5B, bert_base, bert_large</code> or <code>all</code> to run all models.  Note that running all models at once is spatially expensive, requiring ~55GB on disk.  

Results will be logged in <i>output/</i> and in <i>last_results.csv</i>.  