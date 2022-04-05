import logging
import os
import json
import re
import string
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import operator
import rouge
import pandas as pd

class QAData(object):

    def __init__(self, logger, args, data_path, is_training):
        self.data_path = data_path
        self.data_type = data_path.split("/")[-1][:-4]
        dataset_name = data_path.split("/")[0].lower()
        assert self.data_type in ["train", "dev", "test"]

        self.data = {}
        assert data_path.endswith(".tsv"), "data file has to be in tsv format"
        curr_data_path = data_path.replace("{}.tsv".format(self.data_type),
                                           "data_{}_{}.tsv".format(dataset_name, self.data_type))
        self.data = {"id": [], "encoder_inputs": [], "decoder_outputs": []}
        with open(curr_data_path, "r") as f:
            cnt = 0
            for line in f:
                question, answer = line.split("\t")
                question, passage = question.split('\\n')
                input_text = args.encoder_pattern.replace('[Question]', question)
                input_text = input_text.replace('[Passage]', passage)
                self.data["encoder_inputs"].append(input_text)

                decoder_outputs = args.decoder_pattern.replace('[Question]', question)
                decoder_outputs = decoder_outputs.replace('[Passage]', passage)
                decoder_outputs = decoder_outputs.replace('[Answer]', answer)
                self.data["decoder_outputs"].append(decoder_outputs)

                self.data["id"].append("{}-{}".format(self.data_type, cnt))
                cnt += 1

        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args
        self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None
        self.cnt = cnt
        self.metric = "Accuracy"

    def __len__(self):
        return len(self.data["answer"])

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).lower()

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def load_dataset(self, tokenizer, few_shot=0):

        self.tokenizer = tokenizer
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        preprocessed_path = os.path.join(
                "/".join(self.data_path.split("/")[:-1]),
                self.data_path.split("/")[-1].replace(".tsv", "{}{}-ep_{}-dp_{}-{}.json".format(
                    "-uncased" if self.args.do_lowercase else "",
                    "-xbos" if self.args.append_another_bos else "",
                    str(self.args.encoder_pattern),
                    str(self.args.decoder_pattern),
                    postfix)))
        if self.load and os.path.exists(preprocessed_path):
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                input_ids, attention_mask, decoder_output_ids, \
                decoder_output_mask, metadata = json.load(f)
        else:
            print("Start tokenizing...")
            metadata, encoder_inputs, decoder_outputs = [], [], []
            metadata.append((len(encoder_inputs), len(encoder_inputs)+len(self.data["encoder_inputs"])))
            encoder_inputs += self.data["encoder_inputs"]
            decoder_outputs += self.data["decoder_outputs"]

            if self.args.do_lowercase:
                encoder_inputs = [txt.lower() for txt in encoder_inputs]
                decoder_outputs = [txt.lower() for txt in decoder_outputs]

            if self.args.append_another_bos:
                encoder_inputs = ["<s> "+txt for txt in encoder_inputs]
                decoder_outputs = ["<s> " +txt for txt in decoder_outputs]

            encoder_input = self.tokenizer.batch_encode_plus(encoder_inputs,
                                                            padding='max_length', truncation=True,
                                                            max_length=self.args.max_input_length)
            decoder_output = self.tokenizer.batch_encode_plus(decoder_outputs, truncation=True,
                                                            padding='max_length',
                                                            max_length=self.args.max_output_length)
            input_ids, attention_mask = encoder_input["input_ids"], encoder_input["attention_mask"]
            decoder_output_ids, decoder_output_mask = decoder_output["input_ids"], decoder_output["attention_mask"]

            print ("Finish tokenizering...")
            if self.load:
                with open(preprocessed_path, "w") as f:
                    json.dump([input_ids, attention_mask,
                               decoder_output_ids, decoder_output_mask, metadata], f)

        if few_shot:
            if not os.path.isfile('./few_shot_'+str(few_shot) + '.npy'):
                index = random.sample(range(self.cnt), few_shot)
                np.save('./few_shot_'+str(few_shot)+'.npy', np.array(index))
                self.logger.info('Creating index for few shot ' + str(few_shot))
            else:
                index = np.load('./few_shot_'+str(few_shot) + '.npy').tolist()
                self.logger.info('Loading index for few shot ' + str(few_shot))
            index_func = operator.itemgetter(*index)
            self.data["encoder_inputs"] = index_func(self.data["encoder_inputs"])
            self.data["decoder_outputs"] = index_func(self.data["decoder_outputs"])
            self.data["id"] = index_func(self.data["id"])
            input_ids = index_func(input_ids)
            attention_mask = index_func(attention_mask)
            decoder_output_ids = index_func(decoder_output_ids)
            decoder_output_mask = index_func(decoder_output_mask)

        self.dataset = MyQADataset(input_ids, attention_mask,
                                          decoder_output_ids, decoder_output_mask, self.is_training)

    def load_dataloader(self, do_return=False):
        if self.is_training:
            batch_size = self.args.train_batch_size
        else:
            batch_size = self.args.predict_batch_size
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions):
        assert len(predictions)==len(self.data["decoder_outputs"])
        em = np.mean([get_exact_match(prediction, gt) for (prediction, gt) \
                      in zip(predictions, self.data["decoder_outputs"])])
        if self.args.verbose:
            self.logger.info("Accuracy = %.2f" % (100*em))
        return em

    def save_predictions(self, predictions):
        assert len(predictions)==len(self.data["decoder_outputs"])
        save_path = os.path.join(self.args.output_dir, "{}predictions.csv".format(self.args.prefix))
        saved_data = {
            'question': self.data["encoder_inputs"],
            'answer': self.data["decoder_outputs"],
            'prediction': predictions
        }
        df_ = pd.DataFrame(saved_data)
        df_.to_csv(save_path, index=0)
        self.logger.info("Saved prediction in {}".format(save_path))


class MyQADataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_output_ids, decoder_output_mask,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_output_ids = torch.LongTensor(decoder_output_ids)
        self.decoder_output_mask = torch.LongTensor(decoder_output_mask)
        self.is_training = is_training
        assert len(self.input_ids)==len(self.attention_mask)==len(self.decoder_output_ids)==len(self.decoder_output_mask)

        self.length = len(self.input_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.is_training:
            return self.input_ids[idx], self.attention_mask[idx]
        else:
            return self.input_ids[idx], self.attention_mask[idx], \
            self.decoder_output_ids[idx], self.decoder_output_mask[idx]


def get_exact_match(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))


def normalize_answer(s):
    def remove_patterns(text):
        return text.replace('the answer is', '')
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(remove_patterns(lower(s)))))

