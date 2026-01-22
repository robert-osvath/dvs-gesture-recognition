import torch
import numpy as np

class FrameCountStats:
    def __init__(self, batch_first: bool = True, min=0, max=0,running_mean=0, histo=dict()):
        self.batch_first = batch_first
        self.running_mean=running_mean
        self.min_frame_count=min
        self.max_frame_count=max
        self.histo=histo

    def __call__(self, batch):
        for x,_ in batch:
#            print("--looking at ",x.shape)
            l=x.shape[1]
            self.max_frame_count = max(self.max_frame_count,l)
            self.min_frame_count = min(self.max_frame_count,l)
            self.running_mean = self.running_mean*0.9+0.1*np.mean([l for (x,_) in  batch])
            self.histo[l]=1+self.histo[l] if l in self.histo else 1
        return batch

    def getMaxFrameCount(self):
        return self.max_frame_count

    def getMinFrameCount(self):
        return self.min_frame_count
    
    def getRunningMeanFrameCount(self):
        return self.running_mean

    def getHistoFrameCount(self):
        return self.histo

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


class PadTensorsUpdated:
    def __init__(self, batch_first: bool = True, expected_frame_count=-1):
        self.batch_first = batch_first
        self.length = expected_frame_count

    def __call__(self, batch):
        samples_output = []
        targets_output = []

        print("BATCH_LENGTH ",len(batch))
        [print(y,x.shape) for x,y in batch]

        length = self.expected_frame_count
        if (length==-1):
            return batch

        for sample, target in batch:
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            if sample.shape[1]<length:
                if sample.is_sparse:
                    sample.sparse_resize_(
                        (shape[0],length-sample.shape[1], sample.shape[2], sample.shape[3]),
                        sample.sparse_dim(),
                        sample.dense_dim(),
                    )
                else:
                    sample = torch.cat(
                        (
                            sample,
                            torch.zeros(
                                (sample.shape[0],length-sample.shape[1],sample.shape[2], sample.shape[3]),
                                device=sample.device
                            ),
                        ),
                        dim=1
                    )
            elif sample.shape[1]>length:
                sample = sample[:,:length,:,:]
                
            #print("AFTER_PROCESSING:", sample.shape)
            samples_output.append(sample)
            targets_output.append(target)

        samples_output = torch.stack(samples_output, 0 if self.batch_first else 1)
        if len(targets_output[0].shape) > 1:
            targets_output = torch.stack(targets_output, 0 if self.batch_first else -1)
        else:
            targets_output = torch.tensor(targets_output, device=targets_output[0].device)
        #print("SAMPLES_OUTPUT SIZE",samples_output.shape)
        return (samples_output, targets_output)
