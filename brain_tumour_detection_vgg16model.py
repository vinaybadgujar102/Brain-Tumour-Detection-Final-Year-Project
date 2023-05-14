import os
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

YES_DIR_PATH = '/brain_tumor_dataset/yes/'
NO_DIR_PATH = '/brain_tumor_dataset/no/'
yes_imgs_name = os.listdir(YES_DIR_PATH)
no_imgs_name = os.listdir(NO_DIR_PATH)

# Test mode is set to false to avoid unwanted testing outputs from the cells
test = False

def crop_image(img):
    # Resize the image to 256x256 pixels
    resized_img = cv2.resize(
        img,
        dsize=(256, 256),
        interpolation=cv2.INTER_CUBIC
    )
    # Convert the image to grayscale
    gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

    # Apply a Gaussian blur to the image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image by Binary Thresholding
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    
    # perform a series of erosions & dilations to remove any small regions of noise
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop
    ADD_PIXELS = 0
    cropped_img = resized_img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                              extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    return cropped_img


yes_imgs_cropped = [crop_image(cv2.imread(YES_DIR_PATH + img_file)) for img_file in yes_imgs_name]
no_imgs_cropped = [crop_image(cv2.imread(NO_DIR_PATH + img_file)) for img_file in no_imgs_name]

orig_imgs = yes_imgs_cropped + no_imgs_cropped
resized_imgs = [cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC) for img in orig_imgs]
X = np.squeeze(resized_imgs)

# normalize data
X = X.astype('float32')
X /= 255

labels_yes = np.full(len(yes_imgs_name), 1)
labels_no = np.full(len(no_imgs_name), 0)

img_labels = np.concatenate([labels_yes, labels_no])
if test:
    print(img_labels.size, img_labels)

yes_imgs = X[:155]
no_imgs = X[155:]
yes_orig_imgs = orig_imgs[:155]
no_orig_imgs = orig_imgs[155:]

x_yes_train = yes_imgs[:124]
x_yes_valid = yes_imgs[124:]
x_yes_orig_valid = yes_orig_imgs

x_no_train = no_imgs[:78]
x_no_valid = no_imgs[78:]
x_no_orig_valid = no_orig_imgs[78:]

x_train = np.concatenate([x_yes_train, x_no_train])
x_valid = np.concatenate([x_yes_valid, x_no_valid])
x_orig_valid = np.concatenate([x_yes_orig_valid, x_no_orig_valid])

# Splitting the dataset labels for the Training set (i.e `y_train`) and Testing set/Validation set (i.e `y_valid`)
yes_labels = img_labels[:155]
no_labels = img_labels[155:]

y_yes_train = yes_labels[:124]
y_yes_valid = yes_labels[124:]

y_no_train = no_labels[:78]
y_no_valid = no_labels[78:]

y_train = np.concatenate([y_yes_train, y_no_train])
y_valid = np.concatenate([y_yes_valid, y_no_valid])

# Load the pre-trained VGG-16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=10, batch_size=32)

# Saving the trained model
model.save('brain_tumor_detection_model_vgg16.h5')

# Saving the trained model
model.save('brain_tumor_detection_model_vgg16.h5')

loss, accuracy = model.evaluate(x_valid, y_valid)
print('Validation Loss:', loss)
print('Validation Accuracy:', accuracy)

# Print the keys in the history dictionary
print(history.history.keys())

# Plotting training and validation accuracy
plt.plot(history.history['acc'])  # Update key based on available metrics
plt.plot(history.history['val_acc'])  # Update key based on available metrics
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

labels = ["No", "Yes"]
y_hat = model.predict(x_valid)
no_of_indices = 15
random_indices = np.random.choice(
    x_valid.shape[0], size=no_of_indices, replace=False)
# Plot a random sample of 15 test images, with their predicted labels and ground truth
figure = plt.figure(figsize=(20, 13))
sub_title = "Random samples of 15 test images, with their predicted labels and ground truth"
figure.suptitle(sub_title, fontsize=20)
for i in range(no_of_indices):
    rand_index = random_indices[i]

    # Display each image
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_valid[rand_index]))

    # Set the title for each image
    prediction_val = y_hat[rand_index][0]
    predict_index = 0 if (prediction_val < 0.5) else 1
    true_index = y_valid[rand_index]
    prediction = labels[predict_index]
    truth = labels[true_index]
    title_color = "blue" if predict_index == true_index else "red"
    ax_title = "Prediction: {} ({:.2f})\nGround Truth: {}".format(
        prediction, prediction_val, truth)
    ax.set_title(ax_title, color=title_color)
plt.show()