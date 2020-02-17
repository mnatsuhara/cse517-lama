from tqdm import tqdm
import argparse
import spacy
import lama.modules.base_connector as base
from connectors import build_model_by_name
from lm_definitions import LMs

CASED_MODELS = [
  LMs["TransformerXL"],
  LMs["Elmo Original"],
  LMs["Elmo Original 5.5B"],
  LMs["BERT Base CASED"],
  LMs["BERT Large CASED"]
]

CASED_COMMON_VOCAB_FILENAME = "pre-trained_language_models/common_vocab_cased.txt"

# The lowercased models are only used in this vocabulary intersection. They are loaded
# from the huggingface cache. Run_experiments hardcodes lowercase false. These are kept
# for consistency with the paper's results
LOWERCASED_MODELS = [
 {
   # "BERT BASE UNCASED"
   "lm": "bert",
   "bert_model_name": "bert-base-uncased",
   "bert_model_dir": None,
   "bert_vocab_name": "vocab.txt"
 },
 {
   # "BERT LARGE UNCASED"
   "lm": "bert",
   "bert_model_name": "bert-large-uncased",
   "bert_model_dir": None,
   "bert_vocab_name": "vocab.txt"
 },
 {
   # "OpenAI GPT"
   "lm": "gpt",
   "gpt_model_dir": None,
   "gpt_model_name": "openai-gpt"
 }
]

LOWERCASED_COMMON_VOCAB_FILENAME = "pre-trained_language_models/common_vocab_lowercased.txt"


def __intersect_vocabularies(lm_arg_dicts, filename):
    vocabularies = []

    for lm_arg_dict in lm_arg_dicts:
        lm_args = argparse.Namespace(**lm_arg_dict)
        print(lm_args)
        model = build_model_by_name(lm_args.lm, lm_args)

        vocabularies.append(model.vocab)
        print(type(model.vocab))

    if len(vocabularies) > 0:
        common_vocab = set(vocabularies[0])
        for vocab in vocabularies:
            common_vocab = common_vocab.intersection(set(vocab))

        # no special symbols in common_vocab
        common_vocab.difference_update(set(base.SPECIAL_SYMBOLS))

        # print any stop words found, then remove them
        from spacy.lang.en.stop_words import STOP_WORDS
        found_stopwords = common_vocab.intersection(STOP_WORDS)
        for stop_word in found_stopwords:
          print(stop_word)
        common_vocab.difference_update(found_stopwords)

        # Convert to a list
        common_vocab = list(common_vocab)

        # remove punctuation and symbols
        nlp = spacy.load('en')
        manual_punctuation = ['(', ')', '.', ',']
        new_common_vocab = []
        for i in tqdm(range(len(common_vocab))):
            word = common_vocab[i]
            doc = nlp(word)
            token = doc[0]

            # Remove "words" that are multiple words or not actually words
            if(len(doc) != 1):
                print(word)
                for idx, tok in enumerate(doc):
                    print("{} - {}".format(idx, tok))
            elif word in manual_punctuation:
                pass
            elif token.pos_ == "PUNCT":
                print("PUNCT: {}".format(word))
            elif token.pos_ == "SYM":
                print("SYM: {}".format(word))
            else:
                new_common_vocab.append(word)
            # print("{} - {}".format(word, token.pos_))
        common_vocab = new_common_vocab

        # store common_vocab to the passed filename
        with open(filename, 'w') as f:
            for item in sorted(common_vocab):
                f.write("{}\n".format(item))


def main():
    # cased version
    __intersect_vocabularies(CASED_MODELS, CASED_COMMON_VOCAB_FILENAME)
    # lowercased version
    __intersect_vocabularies(LOWERCASED_MODELS, LOWERCASED_COMMON_VOCAB_FILENAME)


if __name__ == '__main__':
    main()
