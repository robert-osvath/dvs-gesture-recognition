import torch
import numpy as np
import torch.nn.functional as F

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
    """
    Resizes the temporal dimension (T) of a (C, T, H, W) tensor to `target_frames`.
    - If input T > target: Downsamples (pools) information.
    - If input T < target: Pads information with zeros.
    This ensures every sample has exactly `target_frames`.
    """
    def __init__(self, batch_first: bool = True, target_frames=-1, mode = 'max'):
        self.batch_first = batch_first
        self.length = target_frames
        self.mode = mode 

    def __call__(self, batch):
        samples_output = []
        targets_output = []

        print("BATCH_LENGTH ",len(batch))
        [print(y,x.shape) for x,y in batch]

        if (self.length==-1):
            return batch

        for sample, target in batch:
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            if sample.shape[1]<self.length:
                if sample.is_sparse:
                    sample.sparse_resize_(
                        (sample.shape[0],self.length-sample.shape[1], sample.shape[2], sample.shape[3]),
                        sample.sparse_dim(),
                        sample.dense_dim(),
                    )
                else:
                    sample = torch.cat(
                        (
                            sample,
                            torch.zeros(
                                (sample.shape[0],self.length-sample.shape[1],sample.shape[2], sample.shape[3]),
                                device=sample.device
                            ),
                        ),
                        dim=1
                    )
            elif sample.shape[1]>self.length:
                if self.mode == 'max':
                    sample = F.adaptive_max_pool3d(sample, output_size=(self.length, None, None))
                elif self.mode == 'avg':
                    sample = F.adaptive_avg_pool3d(sample, output_size=(self.length, None, None))
                elif self.mode == 'linear':
                    sample = sample.unsqueeze(0)
                    sample = F.interpolate(sample, 
                                           size=(self.length, sample.shape[3], sample.shape[4]), 
                                           mode='trilinear', 
                                           align_corners=False)
                    sample = sample.squeeze(0)
                elif self.mode == 'random':
                    indices = torch.randperm(sample.shape[1])[:self.length].sort()[0]
                    sample = sample[indices]
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                
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
