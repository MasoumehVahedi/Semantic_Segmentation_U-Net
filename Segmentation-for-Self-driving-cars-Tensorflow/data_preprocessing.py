import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Split data
def split_dataset(img_list, mask_list):
    image_ds_list = tf.data.Dataset.list_files(img_list, shuffle=False)
    mask_ds_list = tf.data.Dataset.list_files(mask_list, shuffle=False)
    # for dir in zip(image_ds_list.take(5), mask_ds_list.take(5)):
    #   print(dir)

    # Creates a constant tensor from a tensor-like object
    img_filename = tf.constant(img_list)
    mask_filename = tf.constant(mask_list)

    # creates a dataset with a separate element for each row of the input tensor
    dataset = tf.data.Dataset.from_tensor_slices((img_filename, mask_filename))
    # for img, mask in dataset:
    #   print(img)
    #   print(mask)
    return dataset


# Preprocess the Data
def path_process(img_dir, mask_dir):
    image = tf.io.read_file(img_dir)
    # Decode a PNG-encoded image to a uint8 or uint16 tensor
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # print(image)

    mask = tf.io.read_file(mask_dir)
    # Decode a PNG-encoded image to a uint8 or uint16 tensor
    mask = tf.image.decode_png(mask, channels=3)
    # reduce_max() is used to find maximum of elements across dimensions of a tensor
    # keepdims: If it’s set to True it will retain the reduced dimension with length 1
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    # print(mask)

    return image, mask


def preprocess_data(image, mask):
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    input_images = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH), method="nearest")
    input_masks = tf.image.resize(mask, (IMG_HEIGHT, IMG_WIDTH), method="nearest")

    # rescale input images
    input_images = input_images / 255.

    return input_images, input_masks


# To get the final split and preprocessed dataset for the model
def transform_dataset(dataset):
    # Transforming dataset items using map()
    image_dataset = dataset.map(path_process)
    preprocess_image_dataset = image_dataset.map(preprocess_data)

    return image_dataset, preprocess_image_dataset


def plot_images(image_list, mask_list):
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


# Show dataset after training
def display_images(img_list):
  plt.figure(figsize=(12, 12))

  title_img = ["Original Image", "Original Mask", "Predicted Mask"]

  for i in range(len(img_list)):
      plt.subplot(1, len(img_list), i+1)
      plt.title(title_img[i])
      plt.imshow(tf.keras.preprocessing.image.array_to_img(img_list[i]))
      plt.axis('off')
  plt.show()

# Create and show the predicted masks

def show_pred_segments(train_dataset=None, num=None, unet_model=None, img_sample=None, mask_sample=None):
    """
    Displays the first image of each of the num batches
    """
    if train_dataset:
      for image, mask in train_dataset.take(num):
        mask_pred = unet_model.predict(image)
        mask_pred = tf.argmax(mask_pred, axis=-1)
        mask_pred = mask_pred[..., tf.newaxis]
        display_images([image[0], mask[0], mask_pred])
    else:
      mask_pred = unet_model.predict(img_sample[tf.newaxis, ...])
      mask_pred = tf.argmax(mask_pred, axis=-1)
      mask_pred = mask_pred[..., tf.newaxis]
      display_images([img_sample, mask_sample, mask_pred])

