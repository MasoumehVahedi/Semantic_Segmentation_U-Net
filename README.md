# Semantic Segmentation
This project comes from Kaggle Competitons called Semantic Segmentation for Self Driving Cars. In this project, to solve the semantic segmentation problem, there has implemented a deep learning model through Fully Convolutional Network (FCN) called UNET. The architectures of these are briefly introduced in the following.

# U-Net 
The U-Net network account for two parts of model. Firstly, a contracting path and secondly an expansive path. To more specific, Fig 1 show the detailed of U-Net model. As you can see in Fig 1, there are several convolution layers which lead to reduce the spatial size of image, while feature information will be increased. With respect to the second part (right side), through up-convolutions, the spatial features are up sampled. At the end of this model, for each class produces an output segmentation.

# Datasets
Semantic Segmentation for Self Driving Cars Dataset : https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge/tasks
 

