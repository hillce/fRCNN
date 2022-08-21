import sys
import time
import typing as t

import numpy as np
import torch
import torch.distributed as dist


def progressBar(
    batchSize: int,
    curIdx: int,
    totSetSize: int,
    lossVal: float,
    tEpoch: float,
    t0: float,
    stepTotal=10
) -> None:
    """Custom progress bar to check training, validation and testing

    Args:
        batchSize (int): batch size
        curIdx (int): current index in epoch
        totSetSize (int): total dataset size
        lossVal (float): current loss value
        tEpoch (float): current time in epoch
        t0 (float): epoch start time
        stepTotal (int, optional): number of steps in progress bar. Defaults to 10.
    """
    idxMax = totSetSize/batchSize
    chunks = np.arange(0,idxMax,idxMax/(stepTotal))
    t1 = time.time()

    for ii,val in enumerate(chunks):
        if curIdx < val:
            progVal = ii
            break
        if curIdx >= chunks[-1]:
            progVal = len(chunks)
    sys.stdout.write("\r" + f"\33[1;37;40m Progress: [{(progVal-1)*'='}>{(stepTotal-progVal)*'.'}] {curIdx*batchSize}/"
                     f"{totSetSize-1}, Current loss = \33[1;33;40m{lossVal:.8f}\33[1;37;40m, "
                     f"Epoch Time: \33[1;33;40m{t1-tEpoch:.3f} s\33[1;37;40m, "
                     f"Total Time Elapsed: \33[1;33;40m{time.strftime('%H:%M:%S',time.gmtime(t1-t0))}\33[1;37;40m")



def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(
    input_dict,
    average=True
) -> t.Dict:
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
