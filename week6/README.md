# Build a Deep Learning Model That Can Remove Noise from Images Using an Autoencoder on MNIST

## Project Description

This project builds a Denoising Autoencoder using TensorFlow and Keras. The model learns to remove artificial noise from MNIST handwritten digit images by taking noisy images as input and reconstructing clean images as output.

## Dataset

* MNIST Handwritten Digits Dataset
* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 pixels

## Tools Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Jupyter Notebook

## Project Steps

1. Load and preprocess the MNIST dataset.
2. Add Gaussian noise to the images.
3. Build and train the Denoising Autoencoder.
4. Generate denoised images.
5. Compare the original, noisy, and reconstructed images.
6. Save the trained model.

## Files

```text
project.ipynb
README.md
requirements.txt
denoising_autoencoder_mnist.keras
```
## Output

The trained autoencoder removes most of the added noise while preserving the handwritten digit structure.
