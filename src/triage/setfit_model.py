import numpy as np

from . import LABELS

DEFAULT_CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256


def fit(issues, checkpoint=DEFAULT_CHECKPOINT, iterations=10, batch_size=16, epochs=1, seed=7, device=None):
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    model = SetFitModel.from_pretrained(checkpoint, labels=list(LABELS))
    model.model_body.max_seq_length = MAX_TOKENS
    if device:
        model.to(device)
    rows = Dataset.from_dict({"text": [issue.text for issue in issues], "label": [LABELS.index(issue.label) for issue in issues]})
    arguments = TrainingArguments(batch_size=batch_size, num_epochs=epochs, num_iterations=iterations, seed=seed, show_progress_bar=False,
                                  report_to="none", save_strategy="no", eval_strategy="no")
    Trainer(model=model, args=arguments, train_dataset=rows).train()
    return model


def predict_scores(model, texts):
    probabilities = np.asarray(model.predict_proba(texts))
    return [dict(zip(LABELS, (float(value) for value in row))) for row in probabilities]
