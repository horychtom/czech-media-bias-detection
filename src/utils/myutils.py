import torch
import gc
import matplotlib.pyplot as plt
import itertools
import numpy as np

from datasets import load_metric


def preprocess_data(data,tokenizer):
    tokenize = lambda data : tokenizer(data['sentence'], truncation=True)
    
    data = data.map(tokenize,batched=True)
    data = data.remove_columns(['sentence'])
    data.set_format("torch")
    return data

def tokenize(tokenizer,data):
    """ for mapping over dataset"""
    return tokenizer(data['sentence'],truncation=True)

def clean_memory():
    """cuda memory gets full while training transformers"""
    gc.collect()
    torch.cuda.empty_cache()


def compute_metrics(model,device,testing_dataloader):
    """computes F1 score over dataset

    Args:
        model (any type): model for evaluation
        device : gpu or cpu on which are tensors mapped
        testing_dataloader (huggingface dataset): self explained

    Returns:
        dict: {"f1":score}
    """
    metric = load_metric("f1")

    model.eval()
    for batch in testing_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        
    return metric.compute(average='micro')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """ convenient for plotting cm

    Args:
        cm (_type_): confusion matrix
        classes : labels for classes
        normalize (bool, optional):normalize to 1? Defaults to False.
        title : Defaults to 'Confusion matrix'.
        cmap (_type_, optional): Style .Defaults to plt.cm.Blues.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')