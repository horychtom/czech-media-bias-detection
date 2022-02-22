
import torch
import gc
from datasets import load_metric

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