""" Contains some utility functions I used """
import torch
import numpy as np

from allennlp.common import Tqdm
from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def pass_trough_lstm(lstm, inp, lengths, batch_first=True, h_0=None, c_0=None):
    ### Especially handy -- saves packing and padding over and over again
    packed = pack_padded_sequence(inp, lengths, batch_first=batch_first, enforce_sorted=False)
    if h_0 is not None:
        out, rest = lstm(packed, (h_0, c_0))
    else:
        out, rest = lstm(packed)
    out, _ = pad_packed_sequence(out, batch_first=batch_first)
    return out, rest

def get_last_tokens(output, lengths):
    output = output[torch.arange(len(lengths)), lengths-1]
    return output

def print_parameter_counts(model, only_trainable=False):
    total = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad == False and only_trainable == True:
            continue
        
        print(name, "->", np.prod(param.size()))
        if i > 0:
            total += np.prod(param.size())
    print(total)

def comb_to_str(combination):
    s = ''+str(combination[0])
    for el in combination[1:]:
        s += '_'+str(el)
    return s

### The two functions below helped for reimplementing Rocktaschel et al's 
### models as having partially trainable embedding turned out impossible
### in AllenNLP. So I hooked a backward hook that manually zeros the
### gradients of words in W2V vocab.

def grad_zero(t, idx):
    t[idx] = t[idx].fill_(0.)
    return t

def re_read_embeddings_from_text_file(file_uri, embedding_dim, vocab, namespace):

    tokens_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}

    with EmbeddingsTextFile(file_uri) as embeddings_file:
        for line in Tqdm.tqdm(embeddings_file):
            token = line.split(" ", 1)[0]
            if token in tokens_to_keep:
                fields = line.rstrip().split(" ")
                if len(fields) - 1 != embedding_dim:
                    continue

                vector = np.asarray(fields[1:], dtype="float32")
                embeddings[token] = vector

    index_to_token = vocab.get_index_to_token_vocabulary(namespace)

    rows_not_to_optimize = []
    for i in range(vocab_size):
        token = index_to_token[i]

        if token in embeddings:
            rows_not_to_optimize.append(i)

    return rows_not_to_optimize
