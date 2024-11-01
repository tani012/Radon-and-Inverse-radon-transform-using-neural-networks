# Efficient Radon Transform and Inverse Reconstruction Using Deep Neural Networks

## Overview
This project implements a **Convolutional Neural Network (CNN)** to perform the **inverse Radon transform**, reconstructing 2D images from sinograms with higher accuracy and noise resilience than traditional Filtered Back Projection (FBP) methods. The approach aims to improve reconstruction quality, handle noise better, and increase computational efficiency, with applications in **medical imaging**, **seismic analysis**, and **non-destructive testing**.

## Objectives
The primary goals of this project are:
- **Radon Transform**: Generate 1D sinograms from 2D synthetic images.
- **CNN Model**: Develop a CNN for robust inverse Radon transformation.
- **Performance Evaluation**: Compare the CNN model with classical FBP on metrics such as MSE, SSIM, and computational efficiency.

## Methodology

### Data Generation
- **Synthetic Image Creation**: Generate simple synthetic images (e.g., circles) to create the dataset.
- **Radon Transform**: Transform each image into a sinogram by projecting it at angles between 0° and 180°.

### CNN Model Design
The CNN model is designed to reconstruct the original 2D image from a sinogram.
- **Architecture**:
  - **Encoder**: Extracts high-level features from the sinogram by reducing its dimensionality through convolutional and pooling layers.
  - **Decoder**: Reconstructs the original image using transposed convolutions and upsampling techniques.
- **Loss Function**: Uses **Mean Squared Error (MSE)** to minimize the difference between reconstructed and original images. Additionally, **Structural Similarity Index (SSIM)** is used to evaluate the perceptual quality of the reconstructed images.

### Training and Validation
- **Training**: Train the model using synthetic sinograms as input and corresponding images as output.
- **Testing**: Evaluate the model on a test set, comparing reconstructed images to the ground truth.

### Evaluation Metrics
- **Mean Squared Error (MSE)**: Measures the pixel-wise difference between reconstructed and original images.
- **Structural Similarity Index (SSIM)**: Evaluates the structural similarity between original and reconstructed images.
- **Computational Efficiency**: Compares the time and resources needed for the CNN-based reconstruction to classical FBP.

## Results
- **Noise Robustness**: The CNN handles noisy sinograms better than the traditional FBP method.
- **Image Quality**: Higher reconstruction quality (SSIM) and lower MSE than FBP, especially in noisy or incomplete data scenarios.
- **Computation Time**: Although training is resource-intensive, the CNN provides real-time inference post-training, making it efficient for repeated applications.

## Applications
This deep learning approach can be applied in various fields, including:
- **Medical Imaging**: Improved image quality in low-dose CT imaging, where noise is a significant factor.
- **Seismic Imaging**: Enhanced reconstruction of noisy and incomplete seismic data.
- **Non-Destructive Testing**: Increased accuracy for sinogram-based testing in industrial inspection processes.

## Future Work
Potential future improvements include:
1. **3D Radon Transform**: Extending the model to work with 3D data, useful in volumetric imaging (e.g., full-body CT).
2. **Sparse Data Handling**: Adapting the network to better reconstruct images from incomplete sinograms.
3. **Real-World Dataset Application**: Applying the model to real-world CT or MRI datasets, optimizing for specific applications.

## Colab Link
You can access and edit the project in [Google Colab](https://colab.research.google.com/drive/1sLeYY8wmJ9_BOyu81I6JphxGDXCWTA7s?usp=drive_link).

## References
- [Radon Transform Theory and Applications](https://example.com/radon_theory)
- [Convolutional Neural Networks for Image Reconstruction](https://example.com/cnn_reconstruction)

---

This README provides a comprehensive overview of the project, its objectives, methods, results, and applications.

