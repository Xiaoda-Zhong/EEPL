import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def EEPLAttention(query, keys, values):
    query = query.squeeze(0)    # 768

    attention_scores = []
    for i in range(0, keys.shape[0]):
        key = keys[i]   # 768
        attention_score = torch.dot(query, key)
        attention_score = attention_score.reshape(1)
        attention_scores.append(attention_score)

    attention_scores = torch.cat(attention_scores, dim=0)
    attention_scores = attention_scores.float()
    scores_softmax = F.softmax(attention_scores, dim=0)

    attention_embs = []
    for j in range(0, values.shape[0]):
        value = values[j]  # 768
        attention_emb = scores_softmax[j]*value
        attention_emb = attention_emb.unsqueeze(0)
        attention_embs.append(attention_emb)


    attention_embs = torch.cat(attention_embs, dim=0)

    attention_embs = torch.sum(attention_embs, 0)
    attention_embs = attention_embs.unsqueeze(0)


    return attention_embs













