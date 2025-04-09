import torch

class PadTensors:
    def __init__(self, batch_first: bool = True):
        self.batch_first = batch_first

    def __call__(self, batch):
        samples_output = []
        targets_output = []

        max_length = max([sample.shape[1] for sample, target in batch])
        for sample, target in batch:
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            if sample.is_sparse:
                sample.sparse_resize_(
                    (sample.shape[0], max_length - sample.shape[1], sample.shape[2], sample.shape[3]),
                    sample.sparse_dim(),
                    sample.dense_dim(),
                )
            else:
                sample = torch.cat(
                    (
                        sample,
                        torch.zeros(
                            (sample.shape[0], max_length - sample.shape[1], sample.shape[2], sample.shape[3]),
                            device=sample.device
                        ),
                    ),
                    dim=1
                )
            samples_output.append(sample)
            targets_output.append(target)

        samples_output = torch.stack(samples_output, 0 if self.batch_first else 1)
        if len(targets_output[0].shape) > 1:
            targets_output = torch.stack(targets_output, 0 if self.batch_first else -1)
        else:
            targets_output = torch.tensor(targets_output, device=targets_output[0].device)
        return (samples_output, targets_output)
    