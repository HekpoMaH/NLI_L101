""" This file was used both for generating stress test results and for MultiNLI tests """
import json
import csv

from tqdm import tqdm

import torch

from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SnliReader

from allennlp.modules.token_embedders.embedding import Embedding

from model import BowmanEtAlSumOfWords, NLIPredictor, ChenEtAlESIM, RocktaschelEtAlConditionalEncoding, RocktaschelEtAlAttention


t = SnliReader()
###.......................Or the vocabulary your model used
vocab = Vocabulary.from_files('./.vocab/multinli_vocab')
emb = Embedding(vocab.get_vocab_size(), 300)

### choose your architecture here
model = RocktaschelEtAlAttention(vocab, emb).to("cuda")

### load from serialised model here
with open('./.serialization_data/C.E. Attention MultiNLI_Adam_32_0.2_0.0003_0.0_True/best.th', 'rb') as f:
    model.load_state_dict(torch.load(f))

p = NLIPredictor(model, t)

answers = []
with open('./.data/multinli_1.0/multinli_0.9_test_mismatched_unlabeled.jsonl', "r") as input_file:
    for line in tqdm(input_file.readlines()):
        json_obj = json.loads(line)

        ### SWAP the comments of the rows below if predicting for Kaggle competition
        ### (the only way to test on MultiNLI is through Kaggle)
        answers.append({**p.predict_json(json_obj), **json_obj})
        # answers.append((json_obj['pairID'], p.predict_json(json_obj)['prediction']))

### Uncomment this for Kaggle results
# with open('./CE_att_kaggle_mismatched .csv', 'w') as csv_file:
#     wrt = csv.writer(csv_file, delimiter=',', quotechar='"')
#     for line in answers:
#         wrt.writerow(line)


with open('./CE_att_stress_test.jsonl', 'w') as output_file:
    json.dump(answers, output_file)
