import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
(already embedded within the keras library)

fashion_mnist = keras.datasets.fashion_mnist



#assign variables 
(train_images, train_labels ), (test_images, test_labels) = fashion_mnist.load_data()


def category(label):

    '''
   Returns clothing category from label index
    Args:
       label (int): label index
    Returns:
        Printing statmenet of the category

    '''
    if label == 0:
        print("The category of the image is T-Shirt")
    if label == 1:
        print("The category of the image is Trouser")
    if label == 2:
        print("The category of the image is Pullover")
    if label == 3:
        print("The category of the image is Dress")
    if label == 4:
        print("The category of the image is Coat")
    if label == 5:
        print("The category of the image is Sandal")
    if label == 6:
        print("The category of the image is Shirt")
    if label == 7:
        print("The category of the image is Sneaker")
    if label == 8:
        print("The category of the image is Bag")
    if label == 9:
        print("The category of the image is Ankle Boot")

#view a training image and its label

img_index = 0
img = train_images[img_index]
plt.imshow(img, cmap='gray') #show image
category(train_labels[img_index]) #show image label (using pre defined function)



#Create the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),   #reduce dimensioanlity of our images
    keras.layers.Dense(128, activation=tf.nn.relu),  #how many neurals we are going to have
    keras.layers.Dense(10, activation=tf.nn.softmax)  #connecting to the 10 output neurons
])

#Compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=32)

#Evaluate the model
model.evaluate(test_images, test_labels)

#make a prediction
predictions = model.predict(test_images[0:5])



#print the predicted label
print("For the first 5 images the predicted labels are the following")
print(np.argmax(predictions, axis=1))
print("\n")



#print the actual label values (in order to check if the predicted labels are correct)

print("For the first 5 images the actual labels are the following:")
print(test_labels[0:5])


#Print the first 5 images

for i in range(0,5):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28,28))
    plt.imshow(test_images[i], cmap='gray')
    plt.show()


#function to resize the image
def infer_prec(img, img_size):
    img = tf.expand_dims(img, -1)       # from 28 x 28 to 28 x 28 x 1
    img = tf.divide(img, 255)           # normalize
    img = tf.image.resize(img,          # resize acc to the input
             [img_size, img_size])
    img = tf.reshape(img,               # reshape to add batch dimension
            [1, img_size, img_size, 1])
    return img

#importing our own image, resizing it and showing it
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('bag.jpg', 0)  # read image as gray scale
img = cv2.bitwise_not(img)
print(img.shape)   # (300, 231)

plt.imshow(img, cmap="gray")
plt.show()

img = infer_prec(img, 28)  # call preprocess function
print(img.shape)   # (1, 28, 28, 1)

#make a prediction
prediction_bag = model.predict(img)

#print the predicted label
category(np.argmax(prediction_bag, axis=1))
print(np.argmax(prediction_bag, axis=1))

img2 = cv2.imread('dress.jpg', 0)
img2 = cv2.bitwise_not(img2)
print(img2.shape)
plt.imshow(img2, cmap="gray")
plt.show()
img2 = infer_prec(img2, 28)
print(img2.shape)

#make a prediction
prediction_dress = model.predict(img2)

#print the predicted label
print(np.argmax(prediction_dress, axis=1))
category(np.argmax(prediction_dress, axis=1))
