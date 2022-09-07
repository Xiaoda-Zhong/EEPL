import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


alltype = []
file = open("datasets_processed/train.json", 'r', encoding='utf-8')
for line in file.readlines():
    dict = json.loads(line)
    alltype.append(dict["event_type"])

alltype = set(alltype)
# print(alltype)
# print(len(alltype))  # 33


embs = []

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

for type in alltype:
    encoded_input = tokenizer(type, return_tensors='pt', add_special_tokens=False)
    encoded_input.to(device)
    output = model(**encoded_input)
    q_emb = output['last_hidden_state']
    if q_emb.size(1) != 1:
        q_emb = q_emb.mean(axis=1)
    else:
        q_emb = q_emb.squeeze(0)

    # print(q_emb.shape)  # torch.Size([1, 768])
    embs.append(q_emb)

embs = torch.cat(embs, dim=0)
embs.to(device)
# print(embs.shape)  # torch.Size([33, 768])
