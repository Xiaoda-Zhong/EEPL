from trainer import EEPLTrainer, compute_metrics
from module.model import EEPLmodel
from module.init_query import embs
from transformers import TrainingArguments
from preprocessing import train_dataset, eval_dataset

eepl = EEPLmodel(q_embs=embs, concatsize=1536, vocab_size=30567)

training_args = TrainingArguments(
    output_dir="outputs",
    overwrite_output_dir=False,
    evaluation_strategy="steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=256,
    max_steps=3000,
    logging_steps=100,
    save_steps=200,
    save_total_limit=3,
    eval_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model='f1-micro',
    remove_unused_columns=False
)

trainer = EEPLTrainer(
    model=eepl,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=None,
    data_collator = None,
    compute_metrics=compute_metrics,
)

trainer.data_collator=None

trainer.train()

trainer.evaluate()