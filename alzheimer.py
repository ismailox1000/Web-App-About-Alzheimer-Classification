#!/usr/bin/env python
# coding: utf-8
'''
ISMAIL OUBAH 

'''
get_ipython().system('pip install split-folders')

import tensorflow as tf 
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# In[14]:

#split folder into the directory and named the file output and inside it create Train, Test, Val folders with ratio 80% 10% 10%
import splitfolders
splitfolders.ratio('C:\\Users\\is-os\Desktop\\Dataset',output='output',seed=1452,ratio=(0.8,0.1,0.1))

IMG_H=128
IMG_W=128
#define paths of our data and emply them to our variables with a batch size of 64 and the specified Weight and Height
train_data=tf.keras.preprocessing.image_dataset_from_directory('./output/train',image_size=(IMG_H,IMG_W),seed=159,batch_size=64)
test_data=tf.keras.preprocessing.image_dataset_from_directory('./output/test',image_size=(IMG_H,IMG_W),seed=159,batch_size=64)
val_data=tf.keras.preprocessing.image_dataset_from_directory('./output/val',image_size=(IMG_H,IMG_W),seed=159,batch_size=64)
#here we take class names and there is 4 we define them to use later in the code
class_name=train_data.class_names
print(class_name)
#plottingt to see if the pictures are good and if they at random because it will help the training 
plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_name[labels[i]])
        plt.axis("off")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
size = [896,64,3200,2240]
ax.bar(class_name,size)
plt.show

#rescaling data to be between 0 and 1 for better performance
rescaledata = tf.keras.Sequential([
   layers.experimental.preprocessing.Rescaling(1./255),
])
# the input shape for CNN is 4 parameters W,H of images , batch size and channels in this case is 3 as RGB
#32 as input and then 3 hidden layers with 64 neuron and dropping randomly 15% of them to prevent any overfitting and optimize the trainig
#padding 'same' because the images are black in the corners 
#finally outut layer is 4 because 4 classes we have , and softmax to give the class that have the bigger probability to be true
batchsize=64
channels=3
INPUT_SHAPE = (batchsize, IMG_H, IMG_W, channels)
model = models.Sequential([
    rescaledata,
    layers.Conv2D(32, kernel_size = (3,3),padding='same', activation='relu', input_shape=INPUT_SHAPE, kernel_initializer="he_normal"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),
    layers.Conv2D(64,  kernel_size = (3,3),padding='same', activation='relu', input_shape=INPUT_SHAPE, kernel_initializer="he_normal"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),
    layers.Conv2D(64, kernel_size = (3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE, kernel_initializer="he_normal"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.20),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax'),
])

model.build(input_shape=INPUT_SHAPE)

model.summary()

#this parameter is for calculating loss and accuracy
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

#fit the model to train
history = model.fit(
    train_data,
    batch_size=batchsize,
    validation_data=val_data,
    verbose=1,
    epochs=100,
)

#evaluate the test data 
scores = model.evaluate(test_data)

history.history['loss'][10:]
accuracy=history.history['accuracy']
val_accuracy=history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

#simplyy ploting the result of our training into a plot for better visualisation
Epochs=100
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(Epochs), accuracy, label='Training Accuracy')
plt.plot(range(Epochs), val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Train & Val Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(Epochs), loss, label='Training Loss')
plt.plot(range(Epochs), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Train & Val Loss')
plt.show()

#since we need to check the model performing well we need to predict multiple data so we need a function to go through data images
import numpy as np
def prediction_function (model, img):
    img_arr = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_arr = tf.expand_dims(img_arr, 0)

    prediction = model.predict(img_arr)

    predicted_class = class_name[np.argmax(prediction[0])]
    legitness = round(100 * (np.max(prediction[0])), 2)
    return predicted_class, legitness

#plot the images using the function above
plt.figure(figsize=(15, 15))
for images, labels in test_data.take(2):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, legitness = prediction_function(model, images[i].numpy())
        actual_class = class_name[labels[i]] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {legitness}%")

#aving the model to use in the future
from tensorflow.keras.models import load_model
model.save('alzheimerCNN.h5')

