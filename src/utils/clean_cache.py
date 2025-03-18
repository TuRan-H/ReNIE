import logging
import torch
import gc


def clean_cache(knn_model=None, **kwargs):
    """
    Clean cache to avoid memory leak.
    This fixes this issue: https://github.com/huggingface/transformers/issues/22801

    Args:
        knn_model: SimCse类及其子类的实例
    """

    if knn_model and getattr(knn_model, 'res', None):
        knn_model.res.noTempMemory()

    logging.info(f"Cleaning GPU memory. Current memory usage: {torch.cuda.memory_allocated()}")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    logging.info(f"GPU memory usage after cleaning: {torch.cuda.memory_allocated()}")