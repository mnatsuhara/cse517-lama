# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pytorch_pretrained_bert import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from lama.modules.base_connector import *
from unidecode import unidecode

class GPT2(Base_Connector):

    def __init__(self, args):
        super().__init__()

        if args.gpt_model_dir is not None:
            # load bert model from file
            gpt_model_name = args.gpt_model_dir
            print("loading Open AI GPT2 model from disk at {}".format(gpt_model_name))
        else:
            # load GPT model from huggingface cache
            #gpt_model_name = args.gpt_model_name
            #print("loading Open AI GPT2 model from huggingface cache at {}".format(gpt_model_name))
            raise NotImplementedError("gpt2 tokenization archive maps don't have gpt2-xl")

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)

        # GPT2 byte-level BPE removes <unk>, but we need to put something there for MASK.
        # TODO: Special Tokens get "index out of range" because the weight tensor has only
        # the words from the vocab. This makes sense because the byte-level BPE ensures there
        # are no unknown words. So instead of adding '<unk>', use a token that's in the vocab
        # but outside the common vocab; the "huggingface transformers" version of this uses eos.
        # TODO: Consider switching from pytorch-pretrained-bert to huggingface transformers.
        # self.tokenizer.set_special_tokens([self.unk_token])
        self.unk_token = GPT2_EOS

        # GPT uses a different way to represent BPE then BERT, and GPT2 adds more changes.
        # In GPT, the final suffixes (tokens that are at the end of a word) are indicated
        # with </w> suffix, while pieces that must be followed (tokens that are not at the
        # end of the word) are written as is. In BERT the prefixes are written as is
        # while the parts that must follow (not be followed!) have '##' prefix.
        # In GPT2, the BPT is truly byte-level, and rather than an end-of-word, they
        # use a beginning-of-word character \u0120 (256 + ' '). Also, because of the
        # byte encoding, there is no <unk>.
        # There is no one-to-one coversion from either GPT form to BERT. But at least
        # we may make pieces that may form a full word look the same.
        # Note that we should be very careful now, because tokenizer.convert_tokens_to_ids
        # won't work with our vocabulary. Also, the model vocab may have both forms of the
        # word (with and without the leading \u0120 prefix), so in that case we must leave
        # the \u0120 there so the form without prefix will have a unique id.
        model_vocab = set(self.tokenizer.decoder.values())
        def convert_word(model_word):
            if model_word.startswith('\u0120'):
                word = model_word[1:]
                if len(word) > 0 and word not in model_vocab:
                    return word
            return model_word

        _, gpt_vocab = zip(*sorted(self.tokenizer.decoder.items()))
        ## self.model_vocab = 
        self.vocab = [convert_word(word) for word in gpt_vocab]
        self._init_inverse_vocab()

        # TODO: See above re: special tokens. We don't use this, but this makes sure it's there
        # self.unk_id = self.tokenizer.special_tokens[self.unk_token]

        # Load pre-trained model (weights)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        self.gpt_model.eval()
        print(self.gpt_model.config)

        # Sanity check.
        assert len(self.vocab) == self.gpt_model.config.vocab_size
        # TODO assert 0 == self.gpt_model.config.n_special

        # TODO: The docs at https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizer
        # say there is supposed to be an eos_token, but there isn't one in the pytorch-pretrained-bert
        # distribution because its GPT2Tokenizer inherits from object, not from PreTrainedTokenizer
        # like the huggingface version does. So use the value from the docs (and vocab.json).
        self.eos_token = GPT2_EOS
        self.eos_id = self.inverse_vocab[self.eos_token]
        self.model_vocab = self.vocab

    def _cuda(self):
        self.gpt_model.cuda()

    def get_id(self, string):
        # The tokenizer splits some words that are in the vocabulary, which is bad if they are labels.
        if string in self.inverse_vocab:
            return [self.inverse_vocab[string]]

        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # indexed_string = self.convert_ids(indexed_string)
        return indexed_string

    def __get_input_tensors(self, sentence_list):
        """Concatenates, tokenize and converts a sentences to model inputs.

        Args:
            sentence_list: A list of strings. The string may contain a special
            [MASK] token.

        Returns:
            A tuple (src_tensor, dst_tensor, masked_indices, tokenized_text).
                src_tensor: torch.LongTensor with shape (seq_len), the input to
                    the new without the last symbol and with EOS prepended.
                dst_tensor: torch.LongTensor with shape (seq_len).
                masked_indices: A list of indices of [MASK] in dst_tensor.
                tokenized_text: A list of token string.
            """
        # Split the sentence by [MASK] and tokenize the chunks independently.
        tokenized_text = []
        masked_indices = []
        for sentence_idx, sentence in enumerate(sentence_list):
            # Some Unicode character codepoints are out of vocabulary for GPT2.
            # TODO: unidecode strips accents and that's not always right.
            for c in [c for c in list(sentence) if c not in self.tokenizer.byte_encoder]:
                sentence = sentence.replace(c, unidecode(c))

            if sentence_idx > 0:
                tokenized_text.append(self.eos_token)
            for chunk_idx, chunk in enumerate(sentence.split('[MASK]')):
                if chunk_idx > 0:
                    masked_indices.append(len(tokenized_text))
                    tokenized_text.append(self.unk_token)
                chunk = chunk.strip()
                if chunk:
                    tokenized_text.extend(self.tokenizer.tokenize(chunk))

        full_indexed_tokens = [
            self.eos_id
        ] + self.tokenizer.convert_tokens_to_ids(tokenized_text)
        full_tokens_tensor = torch.tensor(full_indexed_tokens)
        src_tensor = full_tokens_tensor[:-1]
        dst_tensor = full_tokens_tensor[1:]

        return src_tensor, dst_tensor, masked_indices, tokenized_text

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if try_cuda:
            self.try_cuda()
        src_tensor_list, dst_tensor_list, masked_indices_list, _ = zip(*[
            self.__get_input_tensors(sentences) for sentences in sentences_list
        ])

        src_tensor_batch = torch.nn.utils.rnn.pad_sequence(
            src_tensor_list, batch_first=True)

        # The model uses shared embedding space for tokens and positions. More
        # precisely, the first len(vocab) indices are reserved for words, the
        # last n_special symbols are reserved for special symbols and the rest
        # is used for positions. Softmax and embedding matrices are shared and
        # as result some of output "symbols" correspond to positions. To fix
        # that we have to manually remove logits for positions.
        with torch.no_grad():
            logits, _ = self.gpt_model(src_tensor_batch.to(self._model_device))
            logits = logits[..., :self.gpt_model.config.vocab_size]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()

        token_ids_list = [
            np.array(dst_tensor.numpy()) for dst_tensor in dst_tensor_list
        ]

        return log_probs, token_ids_list, masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):

        if try_cuda:
            self.try_cuda()

        src_tensor_list, dst_tensor_list, masked_indices_list, _ = zip(*[
            self.__get_input_tensors(sentences) for sentences in sentences_list
        ])

        src_tensor_batch = torch.nn.utils.rnn.pad_sequence(
            src_tensor_list, batch_first=True)

        with torch.no_grad():
            output = self.gpt_model.transformer(src_tensor_batch.to(self._model_device))

        # TODO
        sentence_lengths = None
        tokenized_text_list = None

        # As we only return the last layer, [] to have the same format as other models
        return [output], sentence_lengths, tokenized_text_list
