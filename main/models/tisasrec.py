import numpy as np
import torch
import math

# TiSASRec architecture implementation


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_size, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.w_1 = torch.nn.Linear(hidden_size, hidden_size)
        self.w_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.gelu = torch.nn.GELU()
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.w_2(self.dropout1(self.gelu(self.w_1(inputs))))
        outputs = inputs + self.dropout2(outputs) # residual connection
        return outputs


# Using 2 Conv1D for channel wise fusion
class PointWiseFeedForwardAlternate(torch.nn.Module):
    def __init__(self, hidden_size, dropout_rate):

        super(PointWiseFeedForwardAlternate, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.gelu = torch.nn.GELU()
        self.conv2 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.gelu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs # residual connection
        return outputs


class TimeAwareMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, head_num, dropout_rate, device):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.device = device

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        # Attention = softmax(Q x K.T/sqrt(d)) x V
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling, penalize longer sequences
        attn_weights = attn_weights / math.sqrt(K_.shape[-1])

        # reshape attention and time masks
        #! less memory efficient, use expand instead of repeat for head dimension as well
        # time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        # time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])

        S = attn_weights.shape[-1]
        time_mask = time_mask.unsqueeze(0).unsqueeze(-1).expand(self.head_num, -1, -1, S).reshape(-1, S, S)
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)

        paddings = torch.ones(attn_weights.shape) * (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(self.device)

        # replace with padding element if masked
        attn_weights = torch.where(time_mask, paddings, attn_weights)
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforces causality

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs


class TiSASRec(torch.nn.Module):
    def __init__(
        self,
        *,
        num_items,
        num_categories,
        num_metadata,
        max_len,
        time_span,
        hidden_size,
        num_blocks,
        num_heads,
        dropout,
        device,
    ):
        super(TiSASRec, self).__init__()

        self.num_items = num_items
        self.num_categories = num_categories
        self.num_metadata = num_metadata
        self.device = device

        #* embedding layers
        self.item_emb = torch.nn.Embedding(self.num_items+1, hidden_size, padding_idx=0)
        self.abs_pos_K_emb = torch.nn.Embedding(max_len, hidden_size)
        self.abs_pos_V_emb = torch.nn.Embedding(max_len, hidden_size)
        self.time_matrix_K_emb = torch.nn.Embedding(time_span+1, hidden_size)
        self.time_matrix_V_emb = torch.nn.Embedding(time_span+1, hidden_size)

        #* metadata embedding injection
        self.metadata_cat_emb = torch.nn.EmbeddingBag(self.num_categories+1, hidden_size // 2, padding_idx=0)
        self.metadata_num_emb = torch.nn.Linear(num_metadata, hidden_size // 2)
        self.fusion = torch.nn.Linear(hidden_size + 2 * (hidden_size // 2), hidden_size)
        self.emb_layernorm = torch.nn.LayerNorm(hidden_size)

        #* dropout layers
        self.item_emb_dropout = torch.nn.Dropout(p=dropout)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=dropout)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=dropout)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=dropout)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=dropout)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(
                hidden_size,
                num_heads,
                dropout,
                device,
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_size, dropout)
            self.forward_layers.append(new_fwd_layer)

        self.clear_padding_item_embedding()

    def clear_padding_item_embedding(self):
        if self.item_emb.padding_idx is not None:
            with torch.no_grad():
                self.item_emb.weight[self.item_emb.padding_idx].fill_(0.0)

    def seq2vec(
        self,
        *,
        input_ids,
        metadata_seq,
        category_seq,
        embed_only=False,
    ):
        item_vecs = self.item_emb(input_ids)
        metadata_cat_vecs = self.metadata_cat_emb(category_seq)
        metadata_num_vecs = self.metadata_num_emb(metadata_seq)

        item_vecs *= math.sqrt(self.item_emb.embedding_dim) # boost magnitude of item sequence embedding

        combined = torch.cat([
            item_vecs,
            metadata_num_vecs,
            metadata_cat_vecs,
        ], dim=-1)

        vecs = self.fusion(combined)

        if not embed_only:
            vecs = self.item_emb_dropout(vecs)

        vecs = self.emb_layernorm(vecs)
        return vecs

    def vec2feats(
        self,
        *,
        vecs,
        input_ids,
        time_matrix,
    ):
        #! less efficient
        # positions = np.tile(np.array(range(seq_len)), [batch_size, 1])

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len).expand(batch_size, seq_len)

        # positions = torch.LongTensor(positions).to(self.device)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        # time_matrix = torch.LongTensor(time_matrix).to(self.device)
        time_matrix_K = self.time_matrix_K_emb(time_matrix)
        time_matrix_V = self.time_matrix_V_emb(time_matrix)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # mask padding items (0th item in vocabulary) in input_ids
        timeline_mask = torch.BoolTensor(input_ids == 0).to(self.device)
        vecs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = vecs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(vecs), K=V=vecs

            Q = self.attention_layernorms[i](vecs)
            mha_outputs = self.attention_layers[i](
                Q,
                vecs,
                timeline_mask,
                attention_mask,
                time_matrix_K,
                time_matrix_V,
                abs_pos_K,
                abs_pos_V,
            )
            vecs = Q + mha_outputs

            vecs = self.forward_layernorms[i](vecs)
            vecs = self.forward_layers[i](vecs)
            vecs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(vecs)
        return log_feats


    def encode(
        self,
        *,
        input_ids,
        metadata_seq,
        category_seq,
        time_matrix,
    ):
        return self.vec2feats(
            vecs=self.seq2vec(
                input_ids=input_ids,
                metadata_seq=metadata_seq,
                category_seq=category_seq,
            ),
            input_ids=input_ids,
            time_matrix=time_matrix,
        )


    def training_logits(
        self,
        *,
        input_ids,
        pos_ids,
        neg_ids,
        time_matrix,
        metadata_seq,
        category_seq,
    ): # for training
        log_feats = self.encode(
            input_ids=input_ids,
            metadata_seq=metadata_seq,
            category_seq=category_seq,
            time_matrix=time_matrix,
        )

        pos_embs = self.seq2vec(pos_ids, metadata_seq, category_seq, embed_only=True)
        neg_embs = self.seq2vec(neg_ids, metadata_seq, category_seq, embed_only=True)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits


    #* retrieves embedding of last non-padding element in sequence
    def user_representation(
        self,
        *,
        input_ids,
        metadata_seq,
        category_seq,
        time_matrix,
    ) -> torch.Tensor:
        encoded = self.encode(
            input_ids=input_ids,
            metadata_seq=metadata_seq,
            category_seq=category_seq,
            time_matrix=time_matrix,
        )
        lengths = input_ids.ne(0).sum(dim=1).clamp(min=1) - 1
        return encoded[torch.arange(encoded.size(0), device=input_ids.device), lengths]


    def score_candidates(
        self,
        input_ids,
        candidate_ids,
        time_matrix,
        metadata_seq,
        category_seq,
    ): # for validation
        user_repr = self.user_representation(
            input_ids=input_ids,
            metadata_seq=metadata_seq,
            category_seq=category_seq,
            time_matrix=time_matrix,
        )

        candidate_emb = self.seq2vec(
            input_ids=candidate_ids,
            metadata_seq=metadata_seq,
            category_seq=category_seq,
            embed_only=True,
        )

        return torch.einsum("bd,bkd->bk", user_repr, candidate_emb)

    def score_all_items(
        self,
        *,
        input_ids,
        metadata_seq,
        category_seq,
        time_matrix,
    ) -> torch.Tensor:
        user_repr = self.user_representation(
            input_ids=input_ids,
            metadata_seq=metadata_seq,
            category_seq=category_seq,
            time_matrix=time_matrix,
        )
        all_item_emb = self.item_embedding.weight
        return user_repr @ all_item_emb.transpose(0, 1)


class TiSASRecWithoutMetadata(TiSASRec):
    def __init__(self, *user_args, **kwargs):
        super(TiSASRecWithoutMetadata, self).__init__(*user_args, **kwargs)

    def seq2vec(self, seq, embed_only=False):
        vecs = self.item_emb(seq)

        if not embed_only:
            vecs *= math.sqrt(self.item_emb.embedding_dim) # boost magnitude of item sequence embedding
            vecs = self.item_emb_dropout(vecs)

        return vecs
