from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '/gdrive/My Drive/Colab/대개론_CNN/images_total',
        target_size=(300, 300),
        batch_size=100,
        class_mode='categorical',
        shuffle = True)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
        '/gdrive/My Drive/Colab/대개론_CNN/images2',
        target_size=(300, 300),    
        batch_size=100,
        class_mode='categorical',
        shuffle = True)

import numpy as np
import matplotlib.pyplot as plt
    
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, concatenate
from tensorflow.keras import optimizers
from keras.utils import to_categorical
from keras.datasets import mnist
from tensorflow.keras.models import Model

def plot_loss_curve(history):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()   
    
def inception(x, filter_count):
    con_1 = Conv2D(filters = filter_count[0], kernel_size = (1,1), strides = (1,1), padding = 'same', activation = 'relu')(x)

    con_2 = Conv2D(filters = filter_count[1], kernel_size = (1,1), strides = (1,1), padding = 'same', activation = 'relu')(x)
    con_2 = Conv2D(filters = filter_count[2], kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu')(con_2)
    
    con_3 = Conv2D(filters = filter_count[3], kernel_size = (1,1), strides = (1,1), padding = 'same', activation = 'relu')(x)
    con_3 = Conv2D(filters = filter_count[4], kernel_size = (5,5), strides = (1,1), padding = 'same', activation = 'relu')(con_3)
    
    con_4 = MaxPooling2D(pool_size = (3,3), strides=1, padding = 'same')(x)
    con_4 = Conv2D(filters = filter_count[5], kernel_size = (1,1), strides = (1,1), padding = 'same', activation = 'relu')(con_4)

    ## concat
    x = concatenate([con_1, con_2, con_3, con_4], axis=3)

    return x

def train_mnist_model():

    inputA = Input(shape=(300, 300, 3))

    x = Conv2D(filters = 64, kernel_size = (7,7), strides = (2,2), padding = 'same', activation = 'relu')(inputA)
    x = MaxPooling2D(pool_size = (2,2), strides=2, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters = 64, kernel_size = (1,1), strides = (1,1), padding = 'same', activation = 'relu')(x)
    x = Conv2D(filters = 192, kernel_size = (3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3,3), strides=2, padding = 'same')(x)
    
    x = inception(x, [64, 96, 128, 16, 32, 32])
    x = inception(x, [128, 128, 192, 32, 96, 64])
    x = MaxPooling2D(pool_size = (3,3), strides=2, padding = 'same')(x)
    x = inception(x, [192, 96, 128, 16, 32, 32])
    x = inception(x, [160, 112, 224, 24, 64, 64])
    x = inception(x, [128, 128, 256, 24, 64, 64])
    x = inception(x, [112, 114, 288, 32, 64, 64])
    x = inception(x, [256, 160, 320, 32, 128, 128])
    x = MaxPooling2D(pool_size = (3,3), strides=2, padding = 'same')(x)
    x = inception(x, [256, 160, 320, 32, 128, 128])
    x = inception(x, [384, 192, 384, 48, 128, 128])
    x = MaxPooling2D(pool_size = (7,7), strides=1, padding = 'valid')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    z = Dense(3, activation = 'softmax')(x)
    
    model = Model(inputs=[inputA], outputs=z)

    adam = optimizers.Adam(lr = 0.00001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    model.summary()    
    
    history = model.fit_generator(
                    train_generator,
                    steps_per_epoch=100,
                    epochs=75,
                    validation_data=test_generator,
                    validation_steps=20,
                    workers = 10000)
    
    plot_loss_curve(history.history)
    print(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])    
    
    model.save('/gdrive/My Drive/Colab/대개론_CNN/model-201811514')
    
    return model
  

def predict_image_sample(model, X_test, y_test, test_id=-1):
    if test_id < 0:
        from random import randrange
        test_sample_id = randrange(10000)
    else:
        test_sample_id = test_id
    
    test_image = X_test[test_sample_id]
    
    plt.imshow(test_image, cmap='gray')
    
    test_image = test_image.reshape(1,28,28,1)

    y_actual = y_test[test_sample_id]
    print("y_actual number=", y_actual)
    
    y_pred = model.predict(test_image)
    print("y_pred=", y_pred)
    y_pred = np.argmax(y_pred, axis=1)[0]
    print("y_pred number=", y_pred)
    
if __name__ == '__main__': 
    train_mnist_model()
    #model = load_model('mnist.model')

    #predict_image_sample(model, X_test, y_test)
    
    #for i in range(500):
    #    predict_image_sample(model, X_test, y_test)
