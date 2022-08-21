# fRCNN
fRCNN PyTorch Network for bounding boxes of organs and regions of interest within MRI scans

## Usage Instructions

The folder organ_frcnn contains 3 main scripts:
    - train.py
    - test.py
    - predict.py

### train.py

This script is used to train an fRCNN model. The main entry is by use of the `train_args` function, which can be accessed by running the script directly. At the end of training, the user will have a trained model saved as a `*.pt` file.

### test.py

This script is used to test an fRCNN model. The main entry is by use of the `test_args` function, which can be accessed by running the script directly. This will output a `*.json` file which contains bounding boxe coordinates and classes

### predict.py

This script is used to output a set of `*.png` files for given number of subjects. The main entry is by use of the `predict_args` function, which can be accessed by running the script directly.
