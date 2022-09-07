from transformers import Trainer
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class EEPLTrainer(Trainer):
    def compute_loss(self, model, input, return_outputs=False):
        labels = input['labels'].cpu().numpy()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logits = model(input, labels)

        losslist = []
        for i, label in enumerate(labels):
            logit = logits[i]
            logit = logit.unsqueeze(0)   # 1*V

            target = torch.tensor([label], dtype=torch.long).to(device)

            loss_fct = nn.CrossEntropyLoss()
            single_loss = loss_fct(logit, target)
            single_loss = single_loss.reshape(1)   # 1
            losslist.append(single_loss)

        losslist = torch.cat(losslist, dim=0)    # batch
        loss = torch.mean(losslist)

        return (loss, {'outputs':logits}) if return_outputs else loss


def compute_metrics(eval_pred):
    preds,labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(preds, axis=-1)

    # preds = preds.flatten()
    # labels = labels.flatten()

    return {
        #'accuracy': 100 * ((np.array(preds) == np.array(eval_pred.label_ids)).mean()),
        'f1-macro': 100 * f1_score(labels, preds, average="macro"),
        'f1-micro': 100 * f1_score(labels, preds, average="micro"),
        'recall-macro': 100 * recall_score(labels, preds, average="macro"),
        'recall-micro': 100 * recall_score(labels, preds, average="micro"),
        'precision-macro': 100 * precision_score(labels, preds, average="macro"),
        "precision-micro": 100 * precision_score(labels, preds, average="micro")
    }











            

























