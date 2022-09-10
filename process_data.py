import _pickle as pickle
import jsonlines
from transformers import AutoTokenizer
import torch
import json

def process_train(file):
    fr = open(file, 'rb')  # open的参数是pkl文件的路径
    data = pickle.load(fr)  # 读取pkl文件的内容

    newfile = jsonlines.open("datasets_processed/train.json", "w")

    event_types = ['INJURE', 'DECLARE_BANKRUPTCY', 'SENTENCE', 'TRIAL_HEARING', 'DEMONSTRATE', 'START_POSITION',
                   'TRANSPORT', 'ATTACK', 'EXECUTE', 'DIE', 'END_POSITION', 'NOMINATE', 'TRANSFER_OWNERSHIP', 'FINE',
                   'ACQUIT', 'ARREST_JAIL', 'TRANSFER_MONEY', 'PHONE_WRITE', 'START_ORG', 'BE_BORN', 'SUE', 'CONVICT',
                   'MERGE_ORG', 'END_ORG', 'ELECT', 'MARRY', 'EXTRADITE', 'DIVORCE', 'CHARGE_INDICT', 'MEET',
                   'RELEASE_PAROLE', 'APPEAL', 'PARDON']

    tokenizer = AutoTokenizer.from_pretrained("bert_new")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    for i, sample in enumerate(data):
        newsample = {}
        newsample['text'] = sample['sentence']
        newsample['event_type'] = sample['sentence_type']
        newsample['trigger'] = sample['sentence_keyword']
        newsample['prompt'] = sample['sentence'] + ' ' + '[SEP] The text is talking about' + ' ' + sample['sentence_type'] + '. The trigger is [MASK].'

        trigger_token = tokenizer(newsample['trigger'], return_tensors='pt', add_special_tokens=False).to(device)
        trigger_ids = trigger_token["input_ids"].to(device)  # 1*n
        trigger_ids = trigger_ids.squeeze(0)  # n

        newsample['labels'] = trigger_ids.item()

        for j, type in enumerate(event_types):
            if newsample['event_type'] == type:
                newsample['event_label'] = j

        jsonlines.Writer.write(newfile, newsample)
    newfile.close()


def process_test(file):
    data = []
    fr = open(file, 'r', encoding="utf-8")
    for line in fr.readlines():
        dic = json.loads(line)
        data.append(dic)

    newfile = jsonlines.open("datasets_processed/test.json", "w")

    event_types = ['INJURE', 'DECLARE_BANKRUPTCY', 'SENTENCE', 'TRIAL_HEARING', 'DEMONSTRATE', 'START_POSITION',
                   'TRANSPORT', 'ATTACK', 'EXECUTE', 'DIE', 'END_POSITION', 'NOMINATE', 'TRANSFER_OWNERSHIP', 'FINE',
                   'ACQUIT', 'ARREST_JAIL', 'TRANSFER_MONEY', 'PHONE_WRITE', 'START_ORG', 'BE_BORN', 'SUE', 'CONVICT',
                   'MERGE_ORG', 'END_ORG', 'ELECT', 'MARRY', 'EXTRADITE', 'DIVORCE', 'CHARGE_INDICT', 'MEET',
                   'RELEASE_PAROLE', 'APPEAL', 'PARDON']

    tokenizer = AutoTokenizer.from_pretrained("bert_new")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i, sample in enumerate(data):
        newsample = {}
        newsample['text'] = sample['sentence']
        newsample['event_type'] = sample['sentence_type']
        newsample['trigger'] = sample['sentence_keyword']
        newsample['prompt'] = sample['sentence'] + ' ' + '[SEP] The text is talking about' + ' ' + sample[
            'sentence_type'] + '. The trigger is [MASK].'

        trigger_token = tokenizer(newsample['trigger'], return_tensors='pt', add_special_tokens=False).to(device)
        trigger_ids = trigger_token["input_ids"].to(device)  # 1*n
        trigger_ids = trigger_ids.squeeze(0)  # n

        newsample['labels'] = trigger_ids.item()

        for j, type in enumerate(event_types):
            if newsample['event_type'] == type:
                newsample['event_label'] = j

        jsonlines.Writer.write(newfile, newsample)
    newfile.close()


'''------------processing----------------------'''
process_train('datasets/nyt_train_data.pickle')
process_test('datasets/test_set.json')
