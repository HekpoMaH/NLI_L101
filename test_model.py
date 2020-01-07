""" Script for evaluation """
import torch

from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.iterators.basic_iterator import BasicIterator

from allennlp.modules.token_embedders.embedding import Embedding

from allennlp.training.util import evaluate

from model import BowmanEtAlRNN, BowmanEtAlSumOfWords, NLIPredictor, RocktaschelEtAlConditionalEncoding, RocktaschelEtAlAttention, ChenEtAlESIM

from utils import grad_zero, comb_to_str, re_read_embeddings_from_text_file

t = SnliReader()
### You can choose train/val/test datasets here
train_dataset = t.read('.data/snli_1.0/snli_1.0_train.jsonl')
val_dataset = t.read('.data/snli_1.0/snli_1.0_dev.jsonl')
test_dataset = t.read('.data/snli_1.0/snli_1.0_test.jsonl')

vocab = Vocabulary.from_instances(train_dataset + val_dataset)

vocab = Vocabulary.from_files('./.vocab/snli_vocab')

glove = Embedding(vocab.get_vocab_size(), 300)

### Choose and load model here
model = RocktaschelEtAlAttention(vocab, glove, word_by_word=False).to("cuda")
with open('./.serialization_data/C.E. Attention_Adam_32_0.1_0.0003_5e-05_True/best.th', 'rb') as f:
    model.load_state_dict(torch.load(f))
model.to('cuda')

predictor = NLIPredictor(model=model, dataset_reader=t)

iterator = BasicIterator(batch_size=32)
iterator.index_with(vocab)
final = evaluate(model, train_dataset, iterator, cuda_device=0, batch_weight_key=None)
print(final)
final = evaluate(model, val_dataset, iterator, cuda_device=0, batch_weight_key=None)
print(final)
final = evaluate(model, test_dataset, iterator, cuda_device=0, batch_weight_key=None)
print(final)
