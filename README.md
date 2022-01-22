# NearOOD
This project builds an out-of-distribution (OOD) detection benchmark with a focus on near-OOD testing.

## Summary
The goal of this project is to design a dataset that benchmarks the performance of an out-of-distribution detection framework in the domain of near-OOD detection. The dataset is formed with a basis in computational cognitive psychology, and energy-based OOD detection is tested with this benchmark. Additionally, minor updates to the energy-based framework are proposed and the results are shown.

## Files
- CS762_Final_Project.pdf: The final project report submitted. This report details the motivation, methodlogy, and results of this project.
- HelperMethods.py: This module contains an collection of functions that load, train, and test deep learning models.
- OutDetection.py: This module loads a model and a collection of images, then applies OOD detection algorithms to classify each image as either out- or in-distribution.
- logits.py: This files looks at the logit disitrubtion differences between in-distribution and out-of-distribution datasets to see if any
measurable differences in the distribution could be used to improve the performance of energy-based OOD detection.
- main.py: This module handles the arguement parsin, and calls HelpErMethods.py to train and test a deep learning model. 
