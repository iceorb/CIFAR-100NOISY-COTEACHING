# Overview

The goal of this project was to implement a Coteaching approach using two models to address the challenge of noisy labels. By leveraging the disagreement between the models, a smaller subset of potentially cleaner labels could enhance the robustness of the trained model. 

The model architecture implements a CNN model with multiple convolutional layers and batch normalization. It applies leaky ReLU activation, pooling, and dropout operations to extract features from input images and generate predictions. This model was adapted from the research paper linked below.

![image](https://github.com/icexorb/yonseicv/assets/102640559/205b863a-1ef5-4447-a775-8c533bbf0e07)

## Models
- Simple ResNet

  Attempted to improve performance with label smoothing, dropout, and other regularization techniques - just under 50% test accuracy
- CNN w/ Coteaching

  Achieved similar performance as ResNet model unfortunately. Changes may need to be made to the loss function to account for the added complexity of 100 classes.

  *Addendum, in order to improve validation accuracy, a few things were added-- firstly, data augmentation was mistakenly being performed on the validation set, which is no longer the case. Further, addtl data augmentation was added to help the model generalize even further in order to prevent severe overfit on the training data. In which case, the results of the validation accuracy increased by 12%, and the testing accuracy increased by about 3-4%. 

  ** By reducing the batch size to just 32, we increased the amount of regularization in training-- as well as splitting the data randomly by indices were able to achieve much better validation accuracy. Further hyperparameter tuning would increase the performance of the model in testing accuracy (not shown due to training time, but around 69% accuracy)
  
### Dataset

This model uses a modified version of [CIFAR100-NoisyLabel](https://www.kaggle.com/c/cifar100-image-classification-with-noisy-labels/data)


## Results
![Alt text](/RESNETcurve.png "ResNet results")

Coteaching Model
![Alt text](/results/l_curve_lr1e-05_tau0.4_warmups50_gradual80_epochs150_batch64_2023-09-02_11-37-05.png "Final results 2")


## Thoughts 

The SimpleResNet and other methods (not shown) begin to overfit after just 12 epochs, while data augmentation and other methods may help, it's clear that the data is simply too noisy, too complex, and simply unattainable with other methods.

I propose Coteaching, a method that uses two different models to improve confidence of the models overall.

## References
https://github.com/yeachan-kr/pytorch-coteaching/
https://arxiv.org/pdf/1804.06872.pdf
