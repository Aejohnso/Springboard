# Facial Recognition of Identical Twins using Convolutional Neural Networks

See the final presentation [here](https://github.com/Aejohnso/Springboard/blob/master/Capstone_2_Project/Presentation.pdf).

See the final report [here](https://github.com/Aejohnso/Springboard/blob/master/Capstone_2_Project/Final_Report.pdf).

See the full code [here](https://github.com/Aejohnso/Springboard/blob/master/Capstone_2_Project/Python_Code.ipynb).

## Project Summary

Facial recognition has come a long way in the last few years, thanks to convolutional neural networks (CNNs). The question is, can neural nets tell identical twins apart? I have an identical twin brother, and on a number of occasions I’ve been able to unlock his iPhone with my face. Thus, facial recognition of similar-looking faces, while excellent, is still an important security issue that needs to be addressed.

Beginning with the VGG-Face pretrained model, I removed the last layer and added two Dense layers. The first Dense layer had 100 nodes and the second had 2 nodes. I then trained the CNN to distinguish between me and my identical twin brother. The final testing accuracy was 88%, with an area under the ROC curve of 0.94.

Overall, the model failed in cases with bad or inconsistent lighting or sometimes when we were wearing sunglasses. Note, however, that the model accurately predicted many images with sunglasses. An 88% success rate is still good, at least for identical twins.

### Below is the training data
#### Anthony

![png](TrainingAJ.png)

#### Paul

![png](TrainingPJ.png)

### Below is the training history. The model converged in about 18 epochs.

![png](Training_history.png)

### Below is the ROC curve, for the training, validation, and testing data.

![png](ROC_curve.png)

### False Anthony (model thinks it is Paul)

![png](FalseAJ.png)

### False Paul (model thinks it is Anthony)

![png](FalsePJ.png)
