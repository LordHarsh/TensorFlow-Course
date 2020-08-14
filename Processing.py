import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    model = keras.Sequential([ keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])
    
    ## Compiling the Model
    model.compile( optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    
    ## Evaluate
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    
    ## This pridicts the possiblity of it being from 0 to 9th label in an array
    predictions = model.predict(test_images)
    predictions[0]
    
    ## This returns the label with maximum possiblity
    print(numpy.argmax(predictions[0]))
    
if __name__== "__main__":
    main()