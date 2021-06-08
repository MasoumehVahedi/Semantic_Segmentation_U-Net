import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU

from utils import ImageLoading
from utils import label_encoding
from utils import split_dataset
from utils import categorical_mask
from utils import balance_data
from utils import get_unet_model
from utils import predict_model_on_test_images

################### Set the parameters ########################
path_images = "D:/Self Driving Cars/dataA/dataA/CameraRGB/"
path_masks = "D:/Self Driving Cars/dataA/dataA/CameraSeg/"

image_list = os.listdir(path_images)
mask_list = os.listdir(path_masks)
image_list = [path_images+i for i in image_list]
mask_list = [path_masks+i for i in mask_list]

# The number of classes for segmentation
NUM_CLASSES = 12
# labels = ['Unlabeled','Building','Fence','Other',
#           'Pedestrian', 'Pole', 'Roadline', 'Road',
#           'Sidewalk', 'Vegetation', 'Car','Wall',
#           'Traffic sign']

image_loader = ImageLoading(path_images, path_masks)
images_train = image_loader.resize_images()
masks_train = image_loader.resize_masks()

encoding_reshape_mask_main_shape, encoding_reshape_mask = label_encoding(masks_train)

X_train, X_test, Y_train, Y_test = split_dataset(images_train, encoding_reshape_mask_main_shape)

Y_train_cat, Y_test_cat = categorical_mask(Y_train, Y_test, NUM_CLASSES)

class_weights = balance_data(encoding_reshape_mask)

IMG_HIGTH = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

batch_size = 16
epochs = 25

############# Set the U-Net model ###############
model = get_unet_model()
# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# fit the model
history = model.fit(X_train, Y_train_cat, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test, Y_test_cat),
                    verbose=1,
                    #class_weight=class_weights,    # if the result without class_weights was good, we can comment this parameter
                    shuffle=False)

################ Evaluate the model ###################
_, accuracy = model.evaluate(X_test, Y_test_cat)
print("Accuracy = {} %".format(accuracy * 100.0))

################ save the model #################
model.save('.../multi_unet_model.h5')
print("Model is saved")

############# plot the accuracy and loss ##############
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.plot(epochs, accuracy, 'y', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


################## load the model saved #####################
model.load_weights("UNet_model_25_epochs.h5")

################## Predict the model ########################
y_pred = model.predict(X_test)
# To convert the probability of classes to argmax which find the argument giving the maximum value.
# In fact, argmax gives us the class with the maximum probability
argmax_y_pred = np.argmax(y_pred, axis=3)

################## Keras IoU metric implementing ####################
""" Mean Intersection-Over-Union is a common evaluation metric for semantic image segmentation,
    which first computes the IOU for each semantic class and then computes the average over classes. 
    IOU is defined as follows: IOU = true_positive / (true_positive + false_positive + false_negative). 
    """

IOU = MeanIoU(num_classes=NUM_CLASSES)
IOU.update_state(Y_test[:,:,:,0], argmax_y_pred)
MeanIoU_result = IOU.result().numpy()
print("MeanIoU = {}".format(MeanIoU_result))

# we can get IoU for each class
values = np.array(IOU.get_weights()).reshape(NUM_CLASSES, NUM_CLASSES)
IOU_for_class1 = values[0,0]/(values[0,0] + values[0,1] +values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
IOU_for_class2 = values[1,1]/(values[1,1] + values[1,0] +values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
IOU_for_class3 = values[2,2]/(values[2,2] + values[2,0] +values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
IOU_for_class4 = values[3,3]/(values[3,3] + values[2,0] +values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IOU for class1 = {}".format(IOU_for_class1))
print("IOU for class2 = {}".format(IOU_for_class2))
print("IOU for class3 = {}".format(IOU_for_class3))
print("IOU for class4 = {}".format(IOU_for_class4))

# see the images and mask
plt.imshow(images_train[0,:,:], cmap="gray")
plt.imshow(masks_train[0], cmap="gray")

# We can predict the pre-trained model on other images
predict_model_on_test_images(X_test, Y_test, model)