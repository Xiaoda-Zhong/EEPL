from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
from datasets import load_dataset


train_data = "datasets_processed/train.json"
test_data = "datasets_processed/test.json"


# 定义 Dataset
class EEPLDataset(Dataset):
    def __init__(self, path_to_file, split):
        dataset = load_dataset(path='json', data_files=path_to_file,
                               split=split)

        def f(data):
            return len(data['prompt']) < 300

        self.dataset = dataset.filter(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokenizer = AutoTokenizer.from_pretrained('bert_new')

        text = tokenizer(self.dataset[idx]["text"], padding='max_length', max_length=300, truncation=True, return_tensors='pt')
        prompt = tokenizer(self.dataset[idx]["prompt"], padding='max_length', max_length=300, truncation=True, return_tensors='pt')
        event_type = tokenizer(self.dataset[idx]["event_type"], padding='max_length', max_length=300, truncation=True, return_tensors='pt')
        trigger = tokenizer(self.dataset[idx]["trigger"], padding='max_length', max_length=300, truncation=True, return_tensors='pt')
        event_label = self.dataset[idx]["event_label"]
        labels = self.dataset[idx]["labels"]
        sample = {"text": text, "prompt": prompt, "event_type": event_type, "trigger": trigger, "event_label": event_label, "labels": labels}

        # 返回一个 dict
        return sample


trainset = EEPLDataset(train_data, split='train')
# print(len(trainset))
train_size = int(0.8 * len(trainset))
eval_size = len(trainset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(trainset, [train_size, eval_size])


test_dataset = EEPLDataset(test_data, split='train')
# print(len(test_dataset))

