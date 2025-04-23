Usage
=====

.. _installation:

Installation
------------

The current version of PerturbNet requires Python 3.7. We recommend creating a clean Conda environment using the following command:

.. code-block:: console

   $ conda create -n "PerturbNet" python=3.7

After setting up the environment, you can install the package by running:

.. code-block:: console

   $ conda activate PerturbNet
   $ pip install --upgrade PerturbNet
   
   
We used cuDNN 8.7.0 (cudnn/11.7-v8.7.0) and CUDA 11.7.1 for model training.

We also provide an updated version that removes the dependency on TensorFlow by using Python 3.10. To install:

.. code-block:: console

   $ conda create -n "PerturbNet" python=3.10
   $ conda activate PerturbNet
   $pip install pip install PerturbNet==0.0.3b1



Data and Model Availability
---------------------------

The required data, toy examples, and model weights can be downloaded from `Hugging Face <https://huggingface.co/cyclopeta/PerturbNet_reproduce/tree/main>`_.


