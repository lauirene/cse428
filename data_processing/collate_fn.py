import torch
def collate_fn(batch):
    # Transpose the batch (list of lists) to group elements by position
    batch_transposed = list(zip(*batch))
    
    # Process each position across the batch
    processed_batch = []
    for tensors in batch_transposed:
        if all(t is None for t in tensors):  # If all are None, keep None
            processed_batch.append(None)
        else:  # Otherwise, stack non-None tensors and replace None with zero tensors
            #make sure no None element in the tensors
            any_none = any(t is None for t in tensors)
            assert not any_none, "None element in a list of tensors"

            # ✅ NEW CHECK for sequence_data
            # Check shape of first tensor — if 2D and second dim > 10, it's probably sequence
            if isinstance(tensors[0], torch.Tensor) and tensors[0].ndim == 2 and tensors[0].shape[0] == 4:
                # Sequence data → add extra batch dimension (B, 4, seq_len)
                stacked = [t for t in tensors]
                processed_batch.append(torch.stack(stacked))
            else:
                # Regular tensors → stack normally
                stacked = [t for t in tensors]
                processed_batch.append(torch.stack(stacked))
    
    return processed_batch

