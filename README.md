# Overview

The goal of this project was to implement a Coteaching approach using two models to address the challenge of noisy labels. By leveraging the disagreement between the models, a smaller subset of potentially cleaner labels could enhance the robustness of the trained model. 

The model architecture implements a CNN model with multiple convolutional layers and batch normalization. It applies leaky ReLU activation, pooling, and dropout operations to extract features from input images and generate predictions.

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
![Alt text](/COLEARNINGresults.png "Loss Curve")

Updated Result
![Alt text](/results/l_curve_lr1e-05_tau0.4_warmups50_gradual80_epochs150_batch64_2023-07-31_07-50-31.png "Final results")

Updated result 2
![Alt text](/results/l_curve_lr1e-05_tau0.4_warmups50_gradual80_epochs150_batch64_2023-09-02_11-37-05.png "Final results 2")


## Thoughts 

The model begins to overfit the data after ~12 epochs, data augmentation, such as image transforms and flips may help, label smoothing and weight decay may help.

The primary issue of the model is overfitting the noisy labels, in which traditional methods of regularization don't help immensely. Implementing confidence weighting would help the model train, or revising the Coteaching solution. One idea adjacent to this is to build a self resistance learning model that uses adoption of confident samples to learn.

## References
https://github.com/yeachan-kr/pytorch-coteaching/
