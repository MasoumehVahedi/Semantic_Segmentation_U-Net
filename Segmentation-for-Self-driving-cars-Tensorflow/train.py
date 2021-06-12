import os
import matplotlib.pyplot as plt
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from data_preprocessing import split_dataset
from data_preprocessing import transform_dataset
from data_preprocessing import plot_images
from data_preprocessing import display_images
from data_preprocessing import show_pred_segments
from UNetModel import UNet_model


if __name__ == "__main__":
    path = "..."
    img_dir = os.path.join(path, "./dataA/dataA/CameraRGB/")
    mask_dir = os.path.join(path, "./dataA/dataA/CameraSeg/")
    img_list = os.listdir(img_dir)
    mask_list = os.listdir(mask_dir)
    img_list = [img_dir + i for i in img_list]
    mask_list = [mask_dir + i for i in mask_list]

    epochs = 20
    val_subsplits = 5
    buffer_size = 500
    batch_size = 32

    ######################## Split dataset ##########################
    dataset = split_dataset(img_list, mask_list)

    # To get the final split and preprocessed dataset for the model
    image_dataset, preprocess_image_dataset = transform_dataset(dataset)

    # Show image and the corresponding mask
    plot_images(img_list, mask_list)

    # Show image and mask from image_dataset
    # Show image and mask from image_dataset
    for image, mask in image_dataset.take(1):
        img_sample = image
        mask_sample = mask
        print("Image Shape = {}".format(img_sample.shape))
        print("Mask Shape = {}".format(mask_sample.shape))

    display_images([img_sample, mask_sample])

    # Show image and mask from preprocess_image_dataset
    for image, mask in preprocess_image_dataset.take(1):
        img_sample = image
        mask_sample = mask
        print("Image Shape = {}".format(img_sample.shape))
        print("Mask Shape = {}".format(mask_sample.shape))

    display_images([img_sample, mask_sample])


    ######################## Build the unet model #########################
    unet = UNet_model()
    unet_model = unet.build_unet()
    # Compile the model
    # set loss function
    unet_model.compile(optimizer = "adam",
                       loss = SparseCategoricalCrossentropy(from_logits=True),
                       metrics = ["accuracy"])
    # Train the Model
    preprocess_image_dataset.batch(batch_size)
    dataset_train = preprocess_image_dataset.cache().shuffle(buffer_size).batch(batch_size)
    print(preprocess_image_dataset.element_spec)
    history = unet_model.fit(dataset_train, epochs = epochs)

    ############# plot the accuracy and loss ##############
    loss = history.history["loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    accuracy = history.history['accuracy']
    plt.plot(epochs, accuracy, 'y', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    ################# Show images and mask after training ##################
    for img, mask in image_dataset.take(1):
        img_sample = img
        mask_sample = mask
        print(img_sample.shape)
        print(mask_sample.shape)

    display_images([img_sample, mask_sample])
    # Prediction result
    show_pred_segments(train_dataset=dataset_train,
                       num=12,
                       unet_model=unet_model,
                       img_sample=img_sample,
                       mask_sample=mask_sample)



