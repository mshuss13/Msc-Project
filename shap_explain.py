import torch
import shap
import numpy as np
import matplotlib.pyplot as plt

def compute_shap_values(model, background, test_images):
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)

    # Convert test images to numpy for plotting
    test_numpy = test_images.cpu().numpy()

    # Normalise the image data for visualisation (since they were normalised during training between 0 and 1)
    test_numpy = (test_numpy - test_numpy.min()) / (test_numpy.max() - test_numpy.min())

    # Transpose the axes to match the format expected by matplotlib (height, width, channels)
    test_numpy = np.transpose(test_numpy, (0, 2, 3, 1))

    # Plot the original input images and their SHAP attributions
    shap_numpy = [np.transpose(s, (0, 2, 3, 1)) for s in shap_values]

    # Plot the feature attributions
    shap.image_plot(shap_numpy, test_numpy)
