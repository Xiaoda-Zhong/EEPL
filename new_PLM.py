from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
special_tokens_dict = {'additional_special_tokens': [
  "divorces",
  "extradited",
  "wedded",
  "extraditions",
  "injure",
  "resignations",
  "pardons",
  "suicides",
  "journeying",
  "comed",
  "indictments",
  "extradition",
  "birthed",
  "indict",
  "acquitting",
  "retirements",
  "ceasing",
  "jails",
  "divorcing",
  "indicts",
  "nominating",
  "extradite",
  "bankruptcies",
  "resigns",
  "retires",
  "accuses",
  "acquittals",
  "executes",
  "extraditing",
  "merges",
  "nominates",
  "pardoned",
  "pardoning",
  "acquittal",
  "rallying",
  "tripping",
  "journeyed",
  "summits",
  "indicting",
  "injures",
  "suing",
  "appoints",
  "acquit",
  "electing",
  "sues",
  "indicted",
  "indictment",
  "injured",
  "acquitted"
]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model = AutoModel.from_pretrained("bert-base-uncased")
model.resize_token_embeddings(len(tokenizer))

tokenizer.save_pretrained("bert_new")
model.save_pretrained("bert_new")
