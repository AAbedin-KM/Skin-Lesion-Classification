# Skin-Lesion-Classification
Fine-tuned a pretrained ResNet18 on the HAM10000 dermoscopic image dataset to classify 7 types of skin lesions, achieving 75% test accuracy on an imbalanced medical imaging task.
Made Using PyTorch with dermoscopic images as inputs in which are classified out of 7 types of skin lesions.

# How it is made:
technologies and libraries used: Python , PyTorch , NumPy , Pandas , Matplotlib , sklearn
data is first split between training , testing and validation sets
data is then transformed and loaded via DataLoader,
then models final layer is modified to be a ReLu so a probabilities are returned
then model is then trained on training and validation data 
then test data is then used on model data 
finally, grad-cam is then utilised to output to user what pixels are contributing to the predicted classes the most 


# Optimisations made:
used LR Optimisation to chnage the learning rate based on whenever the current layer the model was on was a fullly connect layer or not to prevent weight changes from beign too drastic ruining the performance of the model 
modified ResNet18s final layer to be a linear output to then make it so that a probabiolity would be produced for all 7 classes with the highest probability being the primary guess of the model 
used 128 batchsize to increase training of model as more data would be loaded in per iteration leading to time for epoch to finish to decrease
used 4 workers to keep the GPU being utilised decreasing GPU idle time. As well as this training time of model is further decreased.

# Results:
grad-cam results:
![gradcampic1](https://github.com/user-attachments/assets/b83fbb51-2875-4543-a61f-ffbe34687a34)
![image](https://github.com/user-attachments/assets/928bfec0-d19f-4261-8407-1adfe1ea3a5c)
![image](https://github.com/user-attachments/assets/1ab339ac-9b92-47b0-a373-5f07da969ab8)

AOC and F1 Scores (from two different runs of the model):
![image](https://github.com/user-attachments/assets/e1edd62e-c823-4d40-85d3-9919fe56518c)
![image](https://github.com/user-attachments/assets/cbdd1b19-86ec-4a04-9def-c84b25d06242)







  
