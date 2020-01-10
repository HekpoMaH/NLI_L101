""" Contains all models """
import torch
import torch.nn as nn

import allennlp.models.model as model

from allennlp.predictors.predictor import Predictor

from allennlp.training.metrics import CategoricalAccuracy

from utils import pass_trough_lstm, get_last_tokens

class FastGRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_first=True, bidirectional=False):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_z = torch.zeros(hidden_dim).to('cuda')
        self.b_h = torch.zeros(hidden_dim).to('cuda')
        self.zeta = torch.randn(1, requires_grad=True).to('cuda')
        self.nu = torch.randn(1, requires_grad=True).to('cuda')
        self.hidden_dim = hidden_dim
    def forward(self, inp, hidden=None):
        batch_size, seq_len, features = inp.shape

        output = torch.zeros(batch_size, seq_len, self.hidden_dim).to('cuda')
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim).to('cuda')

        zeta = torch.sigmoid(self.zeta)
        nu = torch.sigmoid(self.nu)

        for t in range(seq_len):
            inp_t = inp[:, t, :]
            z_t = torch.sigmoid(self.W(inp_t) + self.U(hidden) + self.b_z)
            h_tilde = torch.tanh(self.W(inp_t)+self.U(hidden) + self.b_h)
            hidden = (zeta*(torch.ones_like(z_t) - z_t) + nu) * h_tilde + z_t * hidden
            output[:, t, :] = hidden

        return output, hidden

class NLIBaseClass(model.Model):
    def __init__(self, vocab, embedder, inp_dim=300, hidden_dim=100, use_map=True, p_dropout=0.0, map_is_linear=False):

        super(NLIBaseClass, self).__init__(vocab)
        self.drop = nn.Dropout(p_dropout)
        self.hidden_dim = hidden_dim
        self.embed = embedder

        self.loss = nn.CrossEntropyLoss()
        self.acc = CategoricalAccuracy()

        self.use_map = use_map
        if self.use_map:
            self.map_dim = nn.Linear(inp_dim, hidden_dim)
            if not map_is_linear:
                self.map_dim = nn.Sequential(self.map_dim, nn.Tanh())

        self.output_dict = {}

    def forward(self, premise, hypothesis, label, metadata):
        pass

    def predict(self, premise, hypothesis, label, metadata=None):
        mapping = ['entailment', 'contradiction', 'neutral']
        with torch.no_grad():
            output = self.forward(premise, hypothesis, label, metadata)['class_probs']
            _, idx = torch.max(output, 1)
            return mapping[idx]

    def calculate_output_dict(self, class_probs, label):
        mapping = ['entailment', 'contradiction', 'neutral']

        output_dict = {"class_probs": class_probs}
        _, idx = torch.max(output_dict['class_probs'], 1)
        self.acc(class_probs, label)
        output_dict['loss'] = self.loss(class_probs, label)

        output_dict['prediction'] = [mapping[i] for i in idx.view(-1)]
        output_dict['gold_label'] = [mapping[i] for i in label.view(-1)]

        return output_dict

    def get_metrics(self, reset=False):
        return {"accuracy": self.acc.get_metric(reset)}
    
    def get_lengths(self, premise, hypothesis):
        _, plen = torch.unique(torch.nonzero(premise['tokens'])[:, 0], return_counts=True)
        _, hlen = torch.unique(torch.nonzero(hypothesis['tokens'])[:, 0], return_counts=True)
        return plen, hlen


class BowmanEtAlSumOfWords(NLIBaseClass):
    def __init__(self, vocab, embedder, inp_dim=300, hidden_dim=100, use_map=True, p_dropout=0.0):
        super(BowmanEtAlSumOfWords, self).__init__(vocab, embedder, inp_dim, hidden_dim, use_map, p_dropout)
        self.classifier = BowmanEtAlClassifier(hidden_dim)

    def forward(self, premise, hypothesis, label, metadata):
        prem_embed = self.map_dim(self.embed(premise['tokens']))
        hyp_embed = self.map_dim(self.embed(hypothesis['tokens']))

        prem_embed = self.drop(self.drop(prem_embed).sum(dim=1))
        hyp_embed = self.drop(self.drop(hyp_embed).sum(dim=1))
        
        class_probs = self.classifier(prem_embed, hyp_embed)

        return self.calculate_output_dict(class_probs, label)


class BowmanEtAlClassifier(nn.Module):
    def __init__(self, hidden_dim):
        super(BowmanEtAlClassifier, self).__init__()
        self.classifier = nn.Sequential(
                nn.Sequential(
                    nn.Linear(2*hidden_dim, 200),
                    nn.Tanh()),
                nn.Sequential(
                    nn.Linear(200, 200),
                    nn.Tanh()),
                nn.Sequential(
                    nn.Linear(200, 3),
                    nn.Tanh()))

    def forward(self, premise, hypothesis):
        inp = torch.cat((premise, hypothesis), dim=1)
        return self.classifier(inp)

class BowmanEtAlRNN(NLIBaseClass):
    def __init__(self, vocab, embedder, inp_dim=300, hidden_dim=100, use_map=True, p_dropout=0.0, rnn_type='LSTM'):
        super(BowmanEtAlRNN, self).__init__(vocab, embedder, inp_dim, hidden_dim, use_map, p_dropout)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True) if rnn_type=='LSTM' else nn.RNN(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.classifier = BowmanEtAlClassifier(hidden_dim)

    def forward(self, premise, hypothesis, label, metadata=None):
        plen, hlen = self.get_lengths(premise, hypothesis)

        batch_size = premise['tokens'].shape[0]
        seq_len = premise['tokens'].shape[1]

        prem_embed = self.drop(self.map_dim(self.embed(premise['tokens'])))
        hyp_embed = self.drop(self.map_dim(self.embed(hypothesis['tokens'])))

        prem_rnn, _ = pass_trough_lstm(self.rnn, prem_embed, plen)
        hyp_rnn, _ = pass_trough_lstm(self.rnn, hyp_embed, hlen)

        prem_rnn = self.drop(get_last_tokens(prem_rnn, plen))
        hyp_rnn = self.drop(get_last_tokens(hyp_rnn, hlen))

        class_probs = self.classifier(prem_rnn, hyp_rnn)
        return self.calculate_output_dict(class_probs, label)


class RocktaschelEtAlClassifier(nn.Module):
    def __init__(self, hidden_dim, p_dropout=0.0):
        super(RocktaschelEtAlClassifier, self).__init__()
        self.classifier = nn.Sequential(
                    nn.Dropout(p_dropout),
                    nn.Linear(hidden_dim, 3),
                    nn.Tanh())

    def forward(self, inp):
        return self.classifier(inp)

class RocktaschelEtAlBase(NLIBaseClass):
    def __init__(self, vocab, embedder, inp_dim=300, hidden_dim=100, use_map=True, p_dropout=0.0, rnn_type='LSTM', use_fastgrnn=False):
        super(RocktaschelEtAlBase, self).__init__(vocab, embedder, inp_dim, hidden_dim, use_map, p_dropout, map_is_linear=True)
        self.prem_lstm = FastGRNN(hidden_dim, hidden_dim) if use_fastgrnn else nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.hyp_lstm = FastGRNN(hidden_dim, hidden_dim) if use_fastgrnn else nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.classifier = RocktaschelEtAlClassifier(hidden_dim, p_dropout)

    def forward(self, premise, hypothesis, label, metadata=None):
        pass

class RocktaschelEtAlConditionalEncoding(RocktaschelEtAlBase):

    def __init__(self, vocab, embedder, inp_dim=300, hidden_dim=116, use_map=True, p_dropout=0.0, rnn_type='LSTM', use_fastgrnn=False):
        super(RocktaschelEtAlConditionalEncoding, self).__init__(vocab, embedder, inp_dim, hidden_dim, use_map, p_dropout, use_fastgrnn=use_fastgrnn)
        self.use_fastgrnn = use_fastgrnn

    def forward(self, premise, hypothesis, label, metadata=None):
        plen, hlen = self.get_lengths(premise, hypothesis)

        batch_size = premise['tokens'].shape[0]
        seq_len = premise['tokens'].shape[1]

        prem_embed = self.drop(self.map_dim(self.embed(premise['tokens'])))
        hyp_embed = self.drop(self.map_dim(self.embed(hypothesis['tokens'])))

        if self.use_fastgrnn:
            prem_lstm, c_n = self.prem_lstm(prem_embed)
        else:
            prem_lstm, (h_0, c_n) = pass_trough_lstm(self.prem_lstm, prem_embed, plen)

        prem_lstm = self.drop(prem_lstm)
        h_0 = torch.zeros(1, batch_size, self.hidden_dim, device='cuda')

        if self.use_fastgrnn:
            output, _ = self.hyp_lstm(hyp_embed, c_n)
            output = output[:, -1, :]
        else:
            output, _ = pass_trough_lstm(self.hyp_lstm, hyp_embed, hlen, h_0=h_0, c_0=c_n)
            output = get_last_tokens(output, hlen)

        # classifier has dropout on its input
        class_probs = self.classifier(output)
        return self.calculate_output_dict(class_probs, label)

class RocktaschelEtAlAttention(RocktaschelEtAlBase):

    def __init__(self, vocab, embedder, inp_dim=300, hidden_dim=100, use_map=True, p_dropout=0.0, rnn_type='LSTM', word_by_word=False):
        super(RocktaschelEtAlAttention, self).__init__(vocab, embedder, inp_dim, hidden_dim, use_map, p_dropout)
        self.W_y = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.W_p = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_x = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.word_by_word = word_by_word
        if word_by_word:
            self.W_r = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W_t = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, premise, hypothesis, label, metadata=None):
        def create_repeated(repeated, lengths, seq_len, hidden_dim):
            repeated = torch.repeat_interleave(repeated, seq_len, dim=1)
            mask = torch.zeros_like(repeated)
            mask[:, torch.clamp(lengths+1, 1,  seq_len)-1].fill_(1)
            mask[torch.nonzero((lengths+1)>seq_len).squeeze(), seq_len-1].fill_(0)
            mask = mask.cumsum(dim=1)
            return repeated*(1.-mask)

        def masked_softmax(inp, lengths, seq_len):
            # inp is batch,seqlen,1
            mask = torch.zeros_like(inp)
            mask[:, torch.clamp(lengths+1, 1,  seq_len)-1, 0] = (-1e10)
            mask[torch.nonzero((lengths+1)>seq_len).squeeze(), seq_len-1, 0] = 0
            mask = mask.cumsum(dim=1)
            return torch.softmax(inp+mask, dim=1)

        plen, hlen = self.get_lengths(premise, hypothesis)

        batch_size = premise['tokens'].shape[0]
        seq_len = premise['tokens'].shape[1]

        prem_embed = self.drop(self.map_dim(self.embed(premise['tokens'])))
        hyp_embed = self.drop(self.map_dim(self.embed(hypothesis['tokens'])))

        prem_lstm, (_, c_n) = pass_trough_lstm(self.prem_lstm, prem_embed, plen)
        prem_lstm = self.drop(prem_lstm)
        h_0 = torch.zeros(1, batch_size, self.hidden_dim, device='cuda')

        hyp_lstm, (h_n, _) = pass_trough_lstm(self.hyp_lstm, hyp_embed, hlen, h_0=h_0, c_0=c_n)
        hyp_lstm = self.drop(hyp_lstm)
        h_n = self.drop(h_n.permute(1, 0, 2))

        if not self.word_by_word:
            repeated = create_repeated(self.W_h(h_n), plen, seq_len, prem_lstm.shape[-1])
            
            M = torch.tanh(self.W_y(prem_lstm) + repeated)
            alpha = masked_softmax(self.w(M), plen, seq_len)
            r = torch.matmul(torch.transpose(alpha, 1, 2), prem_lstm).squeeze()


        if self.word_by_word:
            r = torch.zeros(batch_size, self.hidden_dim).cuda()
            rs = torch.zeros_like(hyp_lstm, device='cuda')
            for t in range(hyp_lstm.shape[1]):
                h_t = hyp_lstm[:, t, :].unsqueeze(1)
                repeated = create_repeated(self.W_h(h_t) + self.W_r(r.unsqueeze(1)), plen, seq_len, prem_lstm.shape[-1])
                M_t = torch.tanh(self.W_y(prem_lstm) + repeated)
                alpha = masked_softmax(self.w(M_t), plen, seq_len)
                rhs = torch.tanh(self.W_t(r))
                r = torch.matmul(torch.transpose(alpha, 1, 2), prem_lstm).squeeze() + rhs
                rs[:, t, :] = r
            r = get_last_tokens(rs, hlen)
            
        output = torch.tanh(self.W_p(r) + self.W_x(get_last_tokens(hyp_lstm, hlen)))
        class_probs = self.classifier(output)

        return self.calculate_output_dict(class_probs, label)

class ChenEtAlClassifier(nn.Module):
    def __init__(self, hidden_dim, p_dropout):

        super(ChenEtAlClassifier, self).__init__()
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p_dropout)

        self.F = nn.Sequential(
                nn.Linear(4*2*hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p_dropout))

        self.classifier = nn.Sequential(
                nn.Dropout(p_dropout),
                nn.Linear(4*2*hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(p_dropout),
                nn.Linear(hidden_dim, 3))


    def forward(self, m_prem, m_hyp, premise_lengths, hypothesis_lengths):
        inp_prem = self.F(m_prem)
        inp_hyp = self.F(m_hyp)

        prem_decoded, _ = pass_trough_lstm(self.decoder_lstm, inp_prem, premise_lengths)
        hyp_decoded, _ = pass_trough_lstm(self.decoder_lstm, inp_hyp, hypothesis_lengths)

        prem_avg = prem_decoded.sum(dim=1)/premise_lengths.unsqueeze(1)
        prem_max = prem_decoded.max(dim=1).values

        hyp_avg = hyp_decoded.sum(dim=1)/hypothesis_lengths.unsqueeze(1)
        hyp_max = hyp_decoded.max(dim=1).values

        inp = torch.cat([prem_avg, prem_max, hyp_avg, hyp_max], dim=1)
        return self.classifier(inp)


class ChenEtAlESIM(NLIBaseClass):
    def __init__(self, vocab, embedder, inp_dim=300, hidden_dim=300, use_map=False, p_dropout=0.0):
        super(ChenEtAlESIM, self).__init__(vocab, embedder, inp_dim=inp_dim, hidden_dim=hidden_dim, use_map=False, p_dropout=p_dropout)

        self.encoder_lstm = nn.LSTM(inp_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        self.classifier = ChenEtAlClassifier(hidden_dim, p_dropout)

    def enhance(self, encoded, attended):
        return torch.cat((encoded,
            attended,
            encoded - attended,
            encoded * attended),
            dim=-1)

    def forward(self, premise, hypothesis, label, metadata=None):
        def get_mask(batch_size, plen, hlen, premises, hypotheses):
            premise_dim = torch.max(plen)
            hypothesis_dim = torch.max(hlen)
            mask = torch.ones(batch_size, premise_dim, hypothesis_dim).to('cuda')
            mask[premises[:, :] == 0] = -1e9
            mask[(hypotheses[:, :] == 0).unsqueeze(1).expand(-1, premise_dim, -1)] = -1e9
            return mask

        plen, hlen = self.get_lengths(premise, hypothesis)

        batch_size = premise['tokens'].shape[0]
        mask = get_mask(batch_size, plen, hlen, premise['tokens'], hypothesis['tokens'])

        prem_embed = self.drop(self.embed(premise['tokens']))
        hyp_embed = self.drop(self.embed(hypothesis['tokens']))

        prem_encoded, _ = pass_trough_lstm(self.encoder_lstm, prem_embed, plen)
        hyp_encoded, _ = pass_trough_lstm(self.encoder_lstm, hyp_embed, hlen)
        
        soft_alignment = torch.matmul(prem_encoded, torch.transpose(hyp_encoded, 1, 2))
        soft_alignment *= mask


        prem_weights = torch.softmax(soft_alignment, dim=1)
        hyp_weights = torch.softmax(soft_alignment, dim=0)

        prem_attended = prem_weights.bmm(hyp_encoded)
        hyp_attended = torch.transpose(hyp_weights, 1, 2).bmm(prem_encoded)

        enhanced_prem = self.enhance(prem_encoded, prem_attended)
        enhanced_hyp = self.enhance(hyp_encoded, hyp_attended)

        class_probs = self.classifier(enhanced_prem, enhanced_hyp, plen, hlen)
        return self.calculate_output_dict(class_probs, label)
        

class NLIPredictor(Predictor):
    def _json_to_instance(self, json_dict):
        premise = json_dict['sentence1']
        hypothesis = json_dict['sentence2']
        label = json_dict['gold_label']
        assert label != '-'

        instance = self._dataset_reader.text_to_instance(premise, hypothesis, label)

        return instance
