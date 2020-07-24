# IMPORTS
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# COMMAND LINE ARGUMENTS FOR RUNNING AS SCRIPT
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to input dataset")
ap.add_argument("-p", "--plot", type = str, default = "plot.png", help = "path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type  =str, default = "mask_detector.model", help = "path to output face mask detector model")

args = vars(ap.parse_args())

# MODEL HYPERPARAMETERS
LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 32

# LOADING AND PREPROCESSING IMAGES(TRAINING DATA)
print("Loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2] # so we can get "with_mask" or "without_mask" to get associated with the image

    image = load_img(imagePath, target_size = (224,224))
    image = img_to_array(image)
    image = preprocess_input(image)# scaling pixel intensities to range [-1, 1] for convenience

    data.append(image)
    labels.append(label)

# ENCODING LABELS
data = np.array(data, dtype = "float32")
labels = np.array(labels)

label_bin = LabelBinarizer()
labels = label_bin.fit_transform(labels)
labels = to_categorical(labels)

#SPLITTING DATASET INTO TRAIN AND TEST SETS
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size = 0.20, stratify = labels, random_state = 42)

# DATA AUGMENTATION
augmentation = ImageDataGenerator(rotation_range = 20,
                                    zoom_range = 0.15,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.15,
                                    horizontal_flip = True,
                                    fill_mode = "nearest")

#LOADING IN PRE-TRAINED MobileNetV2 MODEL
base_model = MobileNetV2(weights = "imagenet", include_top = False, input_tensor = Input(shape = (224, 224, 3)))

# DEFINING THE HEAD OF THE MODEL, SUITED TO OUR SPECIFIC CLASSIFICATION PROBLEM
head_model = base_model.output
head_model = AveragePooling2D(pool_size = (7,7))(head_model)
head_model = Flatten(name = 'flatten')(head_model)
head_model = Dense(128, activation = 'relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation = "softmax")(head_model)

model = Model(inputs= base_model.input, outputs =head_model)

for layer in base_model.layers:
    layer.trainable = False

# COMPILING MODEL
print("Compiling model...")

optimizer = Adam(lr = LEARNING_RATE, decay = LEARNING_RATE/EPOCHS)
model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

# TRAINING MODEL
print("Training head...")

head = model.fit(augmentation.flow(X_train, y_train, batch_size = BATCH_SIZE),
                    steps_per_epoch = len(X_train)//BATCH_SIZE,
                    validation_data = (X_test, y_test),
                    validation_steps = len(X_test)//BATCH_SIZE,
                    epochs = EPOCHS)

# EVALUATING MODEL PERFORMANCE ON TEST SET
print("Evaluating network...")
predicted_labels = model.predict(X_test, batch_size = BATCH_SIZE)
predicted_labels = np.argmax(predicted_labels, axis = 1)

print(classification_report(y_test.argmax(axis = 1), predicted_labels, target_names = label_bin.classes_))

# SAVING MODEL
print("Saving mask detector model...")

model.save(args["model"], save_format = "h5")

# PLOTTING TRAINING HISTORY
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), head.history["loss"], label = "Training loss")
plt.plot(np.arange(0, EPOCHS), head.history["val_loss"], label = "Validation loss")
plt.plot(np.arange(0, EPOCHS), head.history["acc"], label = "Training accuracy")
plt.plot(np.arange(0, EPOCHS), head.history["val_acc"], label = "Validation accuracy")
plt.title("TRAINING REPORT: LOSS & ACCURACY")
plt.xlabel("Epoch number")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower_left")
plt.savefig(args["plot"])



# python train_mask_detector.py --dataset dataset
