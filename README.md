# Simple Mask R-CNN Implementation
This project is a little experimental implementation of the
[Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
algorithm for image segmentation. As for the Experiment,
I wanted to find out whether the following priciple actually holds:

    Given a backbone that performs well on classifying my objects
    (on single-object images), that backbone is also suitable for
    detecting and masking my objects (on multi-object images).

My limited resources inspired me to try this principle with a "minimal"
example, i.e. the MNIST dataset.
Therefore, the goal of this project is detection and masking of
scattered MNIST-letters.


## Basis
- Most of the model implementation is inspired by the
  [keras implementation by Matterport](https://github.com/matterport/Mask_RCNN).
- The MNIST backbone was developed based on the
  [keras MNIST example](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py).

## Functionality
### Data Generation
* Crop and mask images: `crop_and_mask_image_files()`
* Load data like the mnist-module: `load_data()`
* Generate random images with labels: `generate_labeled_data_files()`
* Load generated data: `load_labeled_data()`

### Data Parsing/Inspection
* Parse data: `load_backbone_pretraining_data()`, `load_maskrcnn_data()`
* Inspect data: `write_solutions()`