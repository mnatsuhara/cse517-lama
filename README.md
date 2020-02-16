CSE 517 - Natural Language Processing Project
Winter 2020

# Reproducing "Language Models as Knowledge Bases"
The original code base can be found [here](https://github.com/facebookresearch/LAMA), and referenced paper [here](https://www.aclweb.org/anthology/D19-1250.pdf).  

# Models used for comparison
* fairseq-fconv ([reference](http://proceedings.mlr.press/v70/dauphin17a/dauphin17a.pdf))
** Hyperparameters:
*** Number of residual blocks
*** Size of embedding
*** Number of units
*** kernel width
* Transformer-XL ([reference](https://arxiv.org/pdf/1901.02860.pdf))
** Hyperparameters: 
*** Batch size
*** Upper epoch limit
*** Number of tokens to predict (at training time vs. eval time)
*** Length of retained previous heads
*** Total number of layers
*** Number of heads
*** Head dimensions
*** Embedding dimensions
*** Model dimensions
*** Inner dimentions in FF
*** Global drop out rate
*** Attention probability drop out rate
*** Parameter initializer (also for embedding)
*** Optimizer
*** Learning rate
* BERT ([reference](https://arxiv.org/pdf/1810.04805.pdf))
** Hyperparameters:
*** Batch size (training vs. eval)
*** Max sequence length
*** Learning rate (Adam)
*** Max number of masked LM predictions per sequence
*** Number of epochs
*** Number of training and warm-up steps
*** Number of steps per estimator call
*** Max number of eval steps
* ELMo ([reference](https://www.aclweb.org/anthology/N18-1202.pdf))
** Hyperparameters:
*** Drop out rate
*** Learning rate
*** ...
