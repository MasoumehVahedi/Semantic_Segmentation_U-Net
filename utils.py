from model import UNet_model
import os
from glob import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical



# Load all images and masks
class ImageLoading():

  def __init__(self, img_path, mask_path):
    self.img_path = img_path
    self.mask_path = mask_path
    self.IMG_HEIGHT = 128
    self.IMG_WIDTH = 128
    # The number of classes for segmentation
    self.NUM_CLASSES = 12

    # Load images
    self.images_training = self.resize_images()
    print(self.images_training.shape)
    # Load masks
    self.masks_training = self.resize_masks()
    print(self.masks_training.shape)

    # Plot image and mask
    self.plotting = self.plot_images()
    print(self.plotting)

  # resise the image
  def resize_images(self):
    images_training = []

    for dir_path in glob(self.img_path):
      for imagePath in glob(os.path.join(dir_path, "*.png")):
        image = cv2.imread(imagePath, 0)
        image = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT))
        images_training.append(image)
    # Convert to numpy array
    images_training = np.array(images_training)
    return images_training

  # resise the mask
  def resize_masks(self):
    masks_training = []

    for dir_path in glob(self.mask_path):
      for maskPath in glob(os.path.join(dir_path, "*.png")):
        mask = cv2.imread(maskPath, 0)
        mask = cv2.resize(mask, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        masks_training.append(mask)
    # Convert to numpy array
    masks_training = np.array(masks_training)
    return masks_training

  def plot_images(self):
    image_list = os.listdir(self.img_path)
    mask_list = os.listdir(self.mask_path)
    image_list = [self.img_path + i for i in image_list]
    mask_list = [self.mask_path + i for i in mask_list]

    N = 200
    img = cv2.imread(image_list[N])
    mask = cv2.imread(mask_list[N])
    mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0],
                                                                                                           img.shape[1])
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title("RGB Image")
    ax[1].imshow(mask, cmap="Paired")
    ax[1].set_title("Segmentation")


def label_encoding(masks_train):
  # label encoding
  # Moreover, we need to flatten, encode and reshape the multiple dimension array
  encoding = LabelEncoder()   # single vector
  n, h, w = masks_train.shape
  reshape_mask = masks_train.reshape(-1, 1)
  encoding_reshape_mask = encoding.fit_transform(reshape_mask)
  encoding_reshape_mask_main_shape = encoding_reshape_mask.reshape(n, h, w)
  # Se the unique of encoding_reshape_mask_main_shape
  print("encoding_reshape_mask_main_shape = {}".format(np.unique(encoding_reshape_mask_main_shape)))

  return encoding_reshape_mask_main_shape, encoding_reshape_mask



# Split test and train dataset
def split_dataset(images_train, encoding_reshape_mask_main_shape):
  # expand the dimensional of images and masks for training
  images_train = np.expand_dims(images_train, axis=3)
  masks_train_input = np.expand_dims(encoding_reshape_mask_main_shape, axis=3)
  # then normalize image
  images_train = normalize(images_train, axis=1)

  # Split dataset to test and train
  X_train, X_test, Y_train, Y_test = train_test_split(images_train, masks_train_input, test_size=0.1, random_state=42)

  print("The value of classes = {}".format(np.unique(Y_train)))
  return X_train, X_test, Y_train, Y_test



def categorical_mask(Y_train, Y_test, num_classes):
  masks_train_cat = to_categorical(Y_train, num_classes)
  y_train_cat = masks_train_cat.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], num_classes))

  # Do the same for y_test
  masks_test_cat = to_categorical(Y_test, num_classes)
  y_test_cat = masks_test_cat.reshape((Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], num_classes))

  return y_train_cat, y_test_cat



def balance_data(encoding_reshape_mask):
  """
     Now, we want to balance data
     Maybe, there have been some classes with a large area, while others have a small ones,
     so, in that case, we need to balance data using class_weight from sklearn library
  """
  # Array of the classes occurring in the data, as given by np.unique(y_org) with y_org the original class labels
  classes = np.uniqe(encoding_reshape_mask)
  # Array of original class labels per sample
  y = encoding_reshape_mask
  class_weights = class_weight.compute_class_weight("balanced", classes, y)

  print("class weight = {}".format(class_weights))
  return class_weights



# Set the model
def get_unet_model():
  return UNet_model(num_classes=12, IMG_HEIGHT=128, IMG_WIDTH=128, IMG_CHANNELS=3)



def predict_model_on_test_images(X_test, Y_test, model):
  num_images = random.randint(0, len(X_test))
  image_test = X_test[num_images]
  _label = Y_test[num_images]
  image_norm = image_test[:,:,0][:,:,None]
  input_image = np.expand_dims(image_norm, 0)
  pred = (model.predict(input_image))
  img_pred = np.argmax(pred, axis=3)[0,:,:]

  # Plotting images and prediction ones
  plt.figure(figsize=(12,8))
  # Test image
  plt.subplot(231)
  plt.title("Test Image")
  plt.imshow(image_test[:,:,0], cmap="gray")
  # Test Label
  plt.subplot(232)
  plt.title("Test Label")
  plt.imshow(_label[:,:,0], cmap="jet")
  # Prediction on Image
  plt.subplot(233)
  plt.title("Prediction on Image")
  plt.imshow(img_pred, cmap="jet")
  plt.show()