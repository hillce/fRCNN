import time
import sys

import numpy as np


def progressBar(batchSize, curIdx, totSetSize, lossVal, tEpoch, t0, stepTotal=10):
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
