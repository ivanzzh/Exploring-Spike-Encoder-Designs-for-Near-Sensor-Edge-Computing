# Exploring Spike Encoder Designs for Near-Sensor Edge Computing
This project is the code of NICE 2025 paper: Exploring Spike Encoder Designs for Near-Sensor Edge Computing

Requirement:
Python 3.9+
Pytorch 1.10+
Numpy 1.26+

Some older version may also works but we did not test.

Datasets used in paper are given under data folder.
Unzip the data into the dataset folder and update file paths in the code accordingly.

LSTM_baseline.py is the code for LSTM baseline.
General_pop_coding.py is the code for poppulation encoder with backend SNN model.
General_Reserovir.py is the code for reservoir encoder with backend SNN model.
General_Reserovir_comparsion.py s used to verify the necessity of online training.
General_diff_reserovi is the code for comparing the performance in five other reservior structure( Randomly Sparse Connected Reservoir, Recurrent Reservoir, Hierarchical Reservoir, Modular Reservoir, Distance-Constrained Reservoir)


To run the code, just change the corresponding dataset name in code file and simply do:

python3 LSTM_baseline.py

python3 General_pop_coding.py

python3 General_Reserovir.py

python3 General_Reserovir_comparsion.py

or 

python3 General_diff_reserovir.py

result folder is the default folder for execution results.

You can also use other dataset by creating your own dataset class and data loaders.

The dimension of data should be (number_of_samples, sequence_length, channels)


If you use this code or find it helpful in your research, please consider citing the following works:

SOLSA: Neuromorphic Spatiotemporal Online Learning for Synaptic Adaptation
Zhenhang Zhang, Jingang Jin, Haowen Fang, Qinru Qiu. Proceedings of the 2024 29th Asia and South Pacific Design Automation Conference (ASP-DAC), Incheon, South Korea, 2024, pp. 848–853.
DOI: 10.1109/ASP-DAC58780.2024.10473975

Exploring Spike Encoder Designs for Near-Sensor Edge Computing (Accepted)
Jingang Jin, Zhenhang Zhang, Qinru Qiu. Accepted to NICE 2025 (Neuro Inspired Computational Elements Conference).
(To appear)
