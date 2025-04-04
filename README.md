# Exploring Spike Encoder Designs for Near-Sensor Edge Computing
This project is the code of NICE 2025 paper: Exploring Spike Encoder Designs for Near-Sensor Edge Computing

Requirement:
Python 3.9+
Pytorch 1.10+
Numpy 1.26+

Some older version may also works but we did not test.

Datasets used in paper are given under data folder.

baseline.py is the code for LSTM baseline.
popEnc_SNN.py is the code for poppulation encoder with backend SNN model.
resEnc_SNN.py is the code for reservoir encoder with backend SNN model.

To run the code, just change the corresponding dataset name in code file and simply do:

python3 baseline.py

python3 popEnc_SNN.py

or 

python3 resEnc_SNN.py

result folder is the default folder for execution results.

You can also use other dataset by creating your own dataset class and data loaders.

The dimension of data should be (number_of_samples, sequence_length, channels)
