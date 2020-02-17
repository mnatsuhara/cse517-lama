
LMs = {
    "TransformerXL": {
        "lm": "transformerxl",
        "label": "transformerxl",
        "models_names": ["transformerxl"],
        "transformerxl_model_name": "transfo-xl-wt103",
        "transformerxl_model_dir": "pre-trained_language_models/transformerxl/transfo-xl-wt103/",
    },
    "Elmo Original": {
        "lm": "elmo",
        "label": "elmo",
        "models_names": ["elmo"],
        "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway",
        "elmo_vocab_name": "vocab-2016-09-10.txt",
        "elmo_model_dir": "pre-trained_language_models/elmo/original",
        "elmo_warm_up_cycles": 10,
    },
    "Elmo Original 5.5B": {
        "lm": "elmo",
        "label": "elmo5B",
        "models_names": ["elmo"],
        "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway_5.5B",
        "elmo_vocab_name": "vocab-enwiki-news-500000.txt",
        "elmo_model_dir": "pre-trained_language_models/elmo/original5.5B/",
        "elmo_warm_up_cycles": 10,
    },
    "BERT Base CASED": {
        "lm": "bert",
        "label": "bert_base",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12/",
    },
    "BERT Large CASED": {
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16/",
    },
}
