import itertools
import csv

import torch.optim as optim

from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.iterators.basic_iterator import BasicIterator

from allennlp.common.params import Params

from allennlp.modules.token_embedders.embedding import Embedding

from allennlp.training.trainer import Trainer
from allennlp.training.util import evaluate

from model import BowmanEtAlRNN, BowmanEtAlSumOfWords, NLIPredictor, RocktaschelEtAlConditionalEncoding, RocktaschelEtAlAttention, ChenEtAlESIM

from utils import grad_zero, comb_to_str, re_read_embeddings_from_text_file

t = SnliReader()
### Choose datasets here
train_dataset = t.read('.data/snli/snli_1.0/snli_1.0_train.jsonl')
val_dataset = t.read('.data/snli/snli_1.0/snli_1.0_dev.jsonl')
vocab = Vocabulary.from_instances(train_dataset)

### Choose word embeddings. Note it is always trainable - we use a
### backward hook to zero the gradient when we don't optimize
### a part of the word embeddings.
params = Params({
    "pretrained_file": ".vector_cache/glove.840B.300d.txt",
    # "pretrained_file": ".vector_cache/w2v.txt",
    "embedding_dim": 300,
    "trainable": True})
glove = Embedding.from_params(vocab, params)

### NOTE For Rocktaschel et al only, uncomment lines below:
# rows_not_to_optimize = re_read_embeddings_from_text_file('.vector_cache/w2v.txt', 300, vocab, glove._vocab_namespace)
# glove.weight.register_hook(lambda x: grad_zero(x, rows_not_to_optimize))
### NOTE: ENDS HERE


### Choose your hyperparameter search space here
name_csv = ['C.E. Attention']
batch_size_csv = [32]
p_drop_csv = [0, 0.1, 0.2]
lr_csv = [0.0001, 0.0003, 0.001]
l2p_csv = [0, 1e-4, 3e-4, 1e-3]

### ... or if you want particular values, just use 1-element arrays!
# p_drop_csv = [0.2]
# lr_csv = [0.0003]
# l2p_csv = [0*1e-4]

combinations = itertools.product(name_csv, optim_csv, batch_size_csv, p_drop_csv, lr_csv, l2p_csv)
combinations = list(combinations)
total_combinations = len(combinations)

for i, c in enumerate(combinations[:]):

    print('\n\n\n', c)
    print(i, '/', total_combinations)
    name, batch_size, p_drop, lr, L2_penalty = c
    iterator = BasicIterator(batch_size=batch_size)
    iterator.index_with(vocab)

    ### Choose model here
    model = RocktaschelEtAlAttention(vocab, glove,  p_dropout=p_drop, word_by_word=False).to("cuda")
    optimizer = optim.Adam(list(model.parameters())[1:], lr=lr, weight_decay=L2_penalty)
    optimizer.add_param_group({'params': glove.parameters(), 'lr': lr, 'weight_decay': 0})
    print(model)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=val_dataset,
                      patience=15,
                      num_epochs=200,
                      ### When using serialization, best model is
                      ### saved in the serialization directory
                      num_serialized_models_to_keep=6,
                      serialization_dir='./.serialization_data/'+comb_to_str(c),
                      cuda_device=0)
    trainer.train()

    ### If you want to save to a file, uncomment below:
    # final = evaluate(model, val_dataset, iterator, cuda_device=0, batch_weight_key=None)
    # acc = "{:.3f}".format(final['accuracy'] * 100)
    # loss = "{:.5f}".format(final['loss'])
    # line = [name, batch_size, p_drop, lr, L2_penalty, loss, acc]
    # with open('./.validation_results/snli_dataset.csv', mode='a+') as csv_file:
    #     wrt = csv.writer(csv_file, delimiter=',', quotechar='"')
    #     wrt.writerow(line)

vocab.save_to_files('./.vocab/' + name_csv[0] + '_vocab')
