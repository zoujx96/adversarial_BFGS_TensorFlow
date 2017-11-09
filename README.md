# adversarial_BFGS_TensorFlow

Having searched nearly all of the related codes on Github, I could not find even an existing code for generating adversarial examples under TensorFlow in Python. To achieve more convenient use of code for the developers and researchers who are working on adversarial machine learning or related fields, and to seek for more efficient implementation of existing adversarial examples generating algorithms, I developed an API for generating adversarial examples based on BFGS algorithm. The algorithm uses TensorFlow Graph so the code should be run on GPUs. The code could be used on different datasets and different neural network models.

This repository consists of two files: adversarial.py and sourcef.py. The CNN model is constructed in sourcef.py. The training algorithm and the adversarial examples generating algorithm are included in adversarial.py.
