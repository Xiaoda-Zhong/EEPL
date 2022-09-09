import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from module.Attention_module import EEPLAttention
import numpy as np
from module.init_query import embs
import torch.nn.functional as F



class EEPLmodel(nn.Module):
    def __init__(self, q_embs, concatsize, vocab_size):
        super(EEPLmodel, self).__init__()

        self.q_embs = nn.Parameter(q_embs)
        self.PLM = AutoModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(concatsize, vocab_size)
        self.vocab_size = vocab_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data = torch.nn.init.xavier_uniform(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input, labels):
        texts = input['text']
        texts_input_ids = texts['input_ids']
        texts_token_type_ids = texts['token_type_ids']
        texts_attention_mask = texts['attention_mask']

        prompts = input['prompt']
        prompts_input_ids = prompts['input_ids'].cpu()
        prompts_token_type_ids = prompts['token_type_ids']
        prompts_attention_mask = prompts['attention_mask']

        event_labels = input['event_label'].cpu().numpy()

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        mask_pos = np.where(np.array(prompts_input_ids == tokenizer.mask_token_id))
        logits = []

        for i, event_label in enumerate(event_labels):
            q = self.q_embs[event_label]  # 768
            q = q.unsqueeze(0)  # 1*768

            text_input_ids = texts_input_ids[i].to(self.device)
            text_token_type_ids = texts_token_type_ids[i].to(self.device)
            text_attention_mask = texts_attention_mask[i].to(self.device)

            prompt_input_ids = prompts_input_ids[i].to(self.device)
            prompt_token_type_ids = prompts_token_type_ids[i].to(self.device)
            prompt_attention_mask = prompts_attention_mask[i].to(self.device)

            # text_token = tokenizer(text, return_tensors='pt').to(self.device)
            # text_emb = self.PLM(**text_token)
            self.PLM.to(self.device)
            text_emb = self.PLM(input_ids=text_input_ids, token_type_ids=text_token_type_ids, attention_mask=text_attention_mask)
            K = text_emb['last_hidden_state']
            K = K.squeeze(0)   # n*768
            K = K.to(self.device)
            V = K
            V.to(self.device)
            attention_emb = EEPLAttention(q, K, V).to(self.device)   # 1*768

            # prompt_token = tokenizer(prompt, return_tensors='pt').to(self.device)
            # prompt_emb = self.PLM(**prompt_token)
            prompt_emb = self.PLM(input_ids=prompt_input_ids, token_type_ids=prompt_token_type_ids, attention_mask=prompt_attention_mask)
            mask_emb = prompt_emb['last_hidden_state'][0, mask_pos[2][i]]
            mask_emb = mask_emb.unsqueeze(0)   # 1*768
            new_emb = torch.cat([mask_emb, attention_emb], 1).to(self.device)  # 1*1536

            logit = self.fc(new_emb)     # 1*V

            # find the index of text
            index = text_input_ids.to(self.device)  # 1*n
            index = index.squeeze(0)
            indexlist = index.tolist()
            indexlist = [indexlist]


            masklabel = torch.full([1, self.vocab_size], float('-inf')).to(self.device)
            zerot = torch.zeros(1, self.vocab_size).to(self.device)

            masklabel.scatter_(1, torch.LongTensor(indexlist).to(self.device), zerot)   # 1*V

            inputlogit = logit + masklabel

            softmax_logit = F.softmax(inputlogit, dim=1)

            softmax_logit = softmax_logit + masklabel
            
            logits.append(softmax_logit)

        logits = torch.cat(logits, dim=0).to(self.device)   # batch*V


        return logits











