import torch.nn as nn
import torch
import copy
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        if (hidden_size % heads != 0):
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention "
                             "heads (%d)" % (hidden_size, heads))
        self.n_heads = heads
        self.hidden_size = hidden_size
        self.attention_head_size = hidden_size // heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.layernorm = nn.LayerNorm(hidden_size)

    def tanspose_for_socre(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input, attention_mask):
        input_s = input
        q, k, v = input
        mix_query_layer = self.q_linear(q)
        mix_key_layer = self.k_linear(k)
        mix_value_layer = self.q_linear(v)

        query_layer = self.tanspose_for_socre(mix_query_layer)
        key_layer = self.tanspose_for_socre(mix_key_layer)
        value_layer = self.tanspose_for_socre(mix_value_layer)

        attention_score = torch.matmul(query_layer,key_layer.transpose(-1,-2))
        attention_score = attention_score/math.sqrt(self.attention_head_size)
        attention_score = attention_score + attention_mask
        attention_prob = nn.Softmax(dim=-1)(attention_score)
        attention_prob = self.dropout(attention_prob)
        context_layer = torch.matmul(attention_prob,value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.layernorm(context_layer)

        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, hidden_size)
        self.dense_2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input):
        hidden_states = self.dropout(torch.relu(self.dense_1(input)))
        hidden_states = self.dropout(self.dense_2(hidden_states))
        hidden_states = self.layer_norm(hidden_states + input)
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, heads, hidden_size, dropout_prob):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(heads, hidden_size, dropout_prob)
        self.feed_forward = FeedForward(hidden_size, dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feed_forward_output = self.feed_forward(hidden_states)
        return feed_forward_output


class Transformer(nn.Module):
    def __init__(self, layers, heads, hidden_size, dropout_prob):
        super(Transformer, self).__init__()
        layer = TransformerLayer(heads, hidden_size, dropout_prob)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            all_encoder.append(hidden_states)
        return


class SASRec(nn.Module):
    def __init__(self, config):
        super(SASRec, self).__init__()
        # data config:item num\max_length
        # model config:embedding size\self_attention_layer_num\drop_out_prob\heads
        self.n_items = config["item_num"]
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.max_seq_length = config['seq_length']
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.transformer_enc = Transformer(self.n_layers, self.n_heads, self.hidden_size, self.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear and module.bias is not None):
            module.bias.data.zero_()

    def forward(self, data):
        # embedding layer
        item_seq = data['seq']
        item_seq_len = data['seq_length']
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_embedding = self.item_embedding(item_seq)
        item_embedding = self.dropout(item_embedding)
        item_embedding = self.layer_norm(item_embedding + position_embedding)
        # self_attention layer
        attention_mask = self.get_attention_mask(item_seq)
        transform_encoding = self.transformer_enc(item_embedding, attention_mask)
        # predicted layer
        output = transform_encoding[-1]
        output = self.gather_index(output, item_seq_len - 1)
        scores = output @ self.item_embedding.weight.T
        return scores, 0

    def get_attention_mask(self, item_seq):
        """
        matrix_size:[batch_size,seq_len,seq_len]
        Set the top right matrix value to zero
        Generate left-to-right uni-directional attention mask for multi-head attention.
        """
        attention_mask = (item_seq > 0).long()
        extend_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attention_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attention_shape), diagonal=1).to(item_seq.device)
        subsequent_mask = (subsequent_mask == 0).unsuqeeze(1)
        subsequent_mask = subsequent_mask.long()

        extend_attention_mask = extend_attention_mask * subsequent_mask
        extend_attention_mask = extend_attention_mask.to(dtype=next(self.parameters()).dtype)
        extend_attention_mask = (1.0 - extend_attention_mask) * -10000.0
        return extend_attention_mask

    def gather_index(self, output, seq_len):
        seq_len = seq_len.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=seq_len)
        return output_tensor.squeeze(1)
