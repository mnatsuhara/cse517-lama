# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from datetime import datetime
from lama.modules import build_model_by_name
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("model", help="the model you want to run (all, transformerxl, gpt2-xl, elmo, elmo5B, bert_base, bert_large")
args = parser.parse_args()

LMs = {
    'transformerxl' : {
        "lm": "transformerxl",
        "label": "transformerxl",
        "models_names": ["transformerxl"],
        "transformerxl_model_name": "transfo-xl-wt103",
        "transformerxl_model_dir": "pre-trained_language_models/transformerxl/transfo-xl-wt103/",
    },
    'gpt2-xl' : {
        "lm": "gpt2-xl",
        "label": "gpt2-xl",
        "models_names": ["gpt2-xl"],
        "gpt_model_name": "gpt2-xl",
        "gpt_vocab_file": "xl-vocab.json",
        "gpt_merges_file": "xl-merges.json",
        "gpt_model_dir": "pre-trained_language_models/gpt/gpt2-xl/",
    },
    'elmo' : {
        "lm": "elmo",
        "label": "elmo",
        "models_names": ["elmo"],
        "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway",
        "elmo_vocab_name": "vocab-2016-09-10.txt",
        "elmo_model_dir": "pre-trained_language_models/elmo/original",
        "elmo_warm_up_cycles": 10,
    },
    'elmo5B' : {
        "lm": "elmo",
        "label": "elmo5B",
        "models_names": ["elmo"],
        "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway_5.5B",
        "elmo_vocab_name": "vocab-enwiki-news-500000.txt",
        "elmo_model_dir": "pre-trained_language_models/elmo/original5.5B/",
        "elmo_warm_up_cycles": 10,
    },
    'bert_base' : {
        "lm": "bert",
        "label": "bert_base",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12",
    },
    'bert_large' : {
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    # 'roberta' : {
    #     "lm": "roberta",
    #     "label": "roberta",
    #     "models_names": ["roberta"],
    #     "roberta_model_name": "pytorch_model.bin",
    #     "roberta_vocab_name": "vocab.json",
    #     "roberta_model_dir": "pre-trained_language_models/roberta/large/",
    # },
}


def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    all_Precision10 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    # Append to results_file
    results_file = open("last_results.csv", "a", encoding='utf-8')
    results_file.write('\n')

    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": "pre-trained_language_models/common_vocab_cased.txt",
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 32,
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": False,
            "max_sentence_length": 100,
            "threads": -1,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]
            if use_negated_probes:
                PARAMETERS["template_negated"] = relation["template_negated"]

        PARAMETERS.update(input_param)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)
        relation_name = relation["relation"]
        if relation_name == "test":
            relation_name = data_path_pre.replace("/", "") + "_test"

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation_name))
            print("Exception: {}".format(e))
            continue

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        Precision1, Precision10 = run_evaluation(args, shuffle_data=False, model=model)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)
        all_Precision10.append(Precision10)

        results_file.write(
            "[{}] {}: {}, P10 = {}, P1 = {}\n".format(datetime.now(), input_param["label"], relation_name,
                                                      round(Precision10 * 100, 2), round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    mean_p10 = statistics.mean(all_Precision10)
    summaryP1 = "@@@ {} - mean P@10 = {}, mean P@1 = {}".format(input_param["label"], round(mean_p10 * 100, 2), round(mean_p1 * 100, 2))
    print(summaryP1)
    results_file.write(f'{summaryP1}\n')
    results_file.flush()

    for t, l in type_Precision1.items():
        prec1item = f'@@@ Label={input_param["label"]}, type={t}, samples={sum(type_count[t])}, relations={len(type_count[t])}, mean prec1={round(statistics.mean(l) * 100, 2)}\n'
        print (prec1item, flush=True)
        results_file.write(prec1item)
        results_file.flush()

    results_file.close()
    return mean_p1, all_Precision1


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre = "data/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, input_param=ip, use_negated_probes=False)

def run_specific_LM(parameters):
    ip = LMs[args.model]
    print(ip["label"])
    run_experiments(*parameters, input_param=ip, use_negated_probes=False)

if __name__ == "__main__":
    if args.model not in LMs and not(args.model == "all"):
        raise ValueError("Unrecognized Language Model: %s." % args.model)

    print("1. Google-RE")
    parameters = get_GoogleRE_parameters()
    if args.model == "all":
        run_all_LMs(parameters)
    else:
        run_specific_LM(parameters)

    print("2. T-REx")
    parameters = get_TREx_parameters()
    if args.model == "all":
        run_all_LMs(parameters)
    else:
        run_specific_LM(parameters)

    print("3. ConceptNet")
    parameters = get_ConceptNet_parameters()
    if args.model == "all":
        run_all_LMs(parameters)
    else:
        run_specific_LM(parameters)

    print("4. SQuAD")
    parameters = get_Squad_parameters()
    if args.model == "all":
        run_all_LMs(parameters)
    else:
        run_specific_LM(parameters)

