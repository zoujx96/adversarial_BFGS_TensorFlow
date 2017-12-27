# adversarial_BFGS_TensorFlow

This is an efficient adversarial attack API for crafting adversarial examples by L-BFGS algorithm under TensorFlow in Python. The code could be used on MNIST & CIFAR10 datasets. The code uses TensorFlow Graph so it should be run on GPUs. 

This repository consists of two files: adversarial.py and sourcef.py. The CNN model is constructed in sourcef.py. The training algorithm and the adversarial examples crafting algorithm are included in adversarial.py. To implement adversarial attack, you can directly run adversarial.py.
