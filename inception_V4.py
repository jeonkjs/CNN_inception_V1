from keras.preprocessing.image import ImageDataGenerator
#rescale=1./255)
train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '/gdrive/My Drive/Colab/대개론_CNN/images1',
        target_size=(300, 300),
        batch_size=100,
        class_mode='categorical',
        shuffle = True)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
        '/gdrive/My Drive/Colab/대개론_CNN/images2',
        target_size=(300, 300),    
        batch_size=50,
        class_mode='categorical',
        shuffle = True)


import numpy as np
import matplotlib.pyplot as plt
    
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, concatenate, AveragePooling2D, GlobalAveragePooling2D
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
    
def Inception_A(x):
  branch_1 = AveragePooling2D((3, 3), padding='same', strides=1)(x)
  branch_1 = Conv2D(filters = 96, kernel_size = (1, 1), padding='same', strides=1)(branch_1)
    
  branch_2 = Conv2D(filters = 96, kernel_size = (1, 1), padding='same', strides=1)(x)

    
  branch_3 = Conv2D(filters = 64, kernel_size = (1, 1), padding='same', strides=1)(x)
  branch_3 = Conv2D(filters = 96, kernel_size = (3, 3), padding='same', strides=1)(branch_3)
    
  branch_4 = Conv2D(filters = 64, kernel_size = (1, 1), padding='same', strides=1)(x)
  branch_4 = Conv2D(filters = 96, kernel_size = (3, 3), padding='same', strides=1)(branch_4)
  branch_4 = Conv2D(filters = 96, kernel_size = (3, 3), padding='same', strides=1)(branch_4)
    
  x = concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

  return x

def Inception_B(x):
  
  branch_1 = AveragePooling2D((3, 3), padding='same', strides=1)(x)
  branch_1 = Conv2D(filters = 128, kernel_size = (1, 1), padding='same', strides=1)(branch_1)
    
  branch_2 = Conv2D(filters = 384, kernel_size = (1, 1), padding='same', strides=1)(x)
    
  branch_3 = Conv2D(filters = 192, kernel_size = (1, 1), padding='same', strides=1)(x)
  branch_3 = Conv2D(filters = 224, kernel_size = (1, 7), padding='same', strides=1)(branch_3)
  branch_3 = Conv2D(filters = 256, kernel_size = (7, 1), padding='same', strides=1)(branch_3)
    
    
  branch_4 = Conv2D(filters = 192, kernel_size = (1, 1), padding='same', strides=1)(x)
  branch_4 = Conv2D(filters = 192, kernel_size = (1, 7), padding='same', strides=1)(branch_4)
  branch_4 = Conv2D(filters = 224, kernel_size = (7, 1), padding='same', strides=1)(branch_4)
  branch_4 = Conv2D(filters = 224, kernel_size = (1, 7), padding='same', strides=1)(branch_4)
  branch_4 = Conv2D(filters = 256, kernel_size = (7, 1), padding='same', strides=1)(branch_4)
    
  x = concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

  return x

def Inception_C(x):

  branch_1 = AveragePooling2D((3, 3), padding='same', strides=1)(x)
  branch_1 = Conv2D(filters = 256, kernel_size = (1, 1), padding='same', strides=1)(branch_1)
    
  branch_2 = Conv2D(filters = 256, kernel_size = (1, 1), padding='same', strides=1)(x)

  branch_3 = Conv2D(filters = 384, kernel_size = (1, 1), padding='same', strides=1)(x)
  branch_3a = Conv2D(filters = 256, kernel_size = (1, 3), padding='same', strides=1)(branch_3)
  branch_3b = Conv2D(filters = 256, kernel_size = (3, 1), padding='same', strides=1)(branch_3)
    
  branch_4 = Conv2D(filters = 384, kernel_size = (1, 1), padding='same', strides=1)(x)
  branch_4 = Conv2D(filters = 448, kernel_size = (1, 3), padding='same', strides=1)(branch_4)
  branch_4 = Conv2D(filters = 512, kernel_size = (3, 1), padding='same', strides=1)(branch_4)
  branch_4a = Conv2D(filters = 256, kernel_size = (3, 1), padding='same', strides=1)(branch_4)
  branch_4b = Conv2D(filters = 256, kernel_size = (1, 3), padding='same', strides=1)(branch_4)
    
  x = concatenate([branch_1, branch_2, branch_3a, branch_3b, branch_4a, branch_4b], axis=3)
    
  return x

def Reduction_A(x):
  k, l, m, n = 192, 224, 256, 384
  branch_1 = MaxPooling2D((3, 3), padding='valid', strides=2)(x)
  
  branch_2 = Conv2D(filters = n, kernel_size = (3, 3), padding='valid', strides=2)(x)

  branch_3 = Conv2D(filters = k, kernel_size = (1, 1), padding='same', strides=1)(x)
  branch_3 = Conv2D(filters = l, kernel_size = (3, 3), padding='same', strides=1)(branch_3)
  branch_3 = Conv2D(filters = m, kernel_size = (3, 3), padding='valid', strides=2)(branch_3)

  x = concatenate([branch_1, branch_2, branch_3], axis=3)
  
  return x

def Reduction_B(x):
  branch_1 = MaxPooling2D((3, 3), padding='valid', strides=2)(x)
  
  branch_2 = Conv2D(filters = 192, kernel_size = (1, 1), padding='same', strides=1)(x)
  branch_2 = Conv2D(filters = 192, kernel_size = (3, 3), padding='valid', strides=2)(branch_2)

  branch_3 = Conv2D(filters = 256, kernel_size = (1, 1), padding='same', strides=1)(x)
  branch_3 = Conv2D(filters = 256, kernel_size = (1, 7), padding='same', strides=1)(branch_3)
  branch_3 = Conv2D(filters = 320, kernel_size = (7, 1), padding='same', strides=1)(branch_3)
  branch_3 = Conv2D(filters = 320, kernel_size = (3, 3), padding='valid', strides=2)(branch_3)

  x = concatenate([branch_1, branch_2, branch_3], axis=3)

  return x

def train_mnist_model():

# 두 개의 입력층을 정의
    inputA = Input(shape=(300, 300, 3))

    x = Conv2D(filters = 32, kernel_size = (3, 3), padding='valid', strides=2)(inputA) # 299x299x3 -> 149x149x32
    x = Conv2D(filters = 32, kernel_size = (3, 3), padding='valid', strides=1)(x) # 149x149x32 -> 147x147x32
    x = Conv2D(filters = 64, kernel_size = (3, 3), padding='same', strides=1)(x) # 147x147x32 -> 147x147x64
    
        
    branch_1 = MaxPooling2D((3, 3), padding='valid', strides=2)(x)
    branch_2 = Conv2D(filters = 96, kernel_size = (3, 3), padding='valid', strides=2)(x)
    x = concatenate([branch_1, branch_2], axis = 3) # 73x73x160
        
    branch_1 = Conv2D(filters = 64, kernel_size = (1, 1), padding='same', strides=1)(x)
    branch_1 = Conv2D(filters = 96, kernel_size = (3, 3), padding='valid', strides=1)(branch_1)

    branch_2 = Conv2D(filters = 64, kernel_size = (1, 1), padding='same', strides=1)(x)
    branch_2 = Conv2D(filters = 64, kernel_size = (7, 1), padding='same', strides=1)(branch_2)
    branch_2 = Conv2D(filters = 64, kernel_size = (1, 7), padding='same', strides=1)(branch_2)
    branch_2 = Conv2D(filters = 96, kernel_size = (3, 3), padding='valid', strides=1)(branch_2)
    x = concatenate([branch_1, branch_2], axis = 3) # 71x71x192
        
    branch_1 = Conv2D(filters = 192, kernel_size = (3, 3), padding='valid', strides=2)(x) # Fig.4 is wrong
    branch_2 = MaxPooling2D((3, 3), padding='valid', strides=2)(x)
    x = concatenate([branch_1, branch_2], axis = 3) 


    for i in range(4):
      x = Inception_A(x)
    
    x = Reduction_A(x)

    for i in range(7):
      x = Inception_B(x)
    
    x = Reduction_B(x)

    for i in range(3):
      x = Inception_C(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.8)(x)
    z = Dense(3, activation = 'softmax')(x)

    model = Model(inputs=[inputA], outputs=z)

    adam = optimizers.Adam(lr = 0.00005)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

    #모델.요약을 하면 원래 사이즈가 28, 28, 1이었는데 26, 26, 32로 바뀌었다, 커널사이즈가 3이라서 28 -> 26으로 줄었고, 필터 개수가 32라서 마지막차원이 32로 변했다.
    model.summary()    
    
    #X_train과 y_train으로 학습을 하고, 에폭마다 평가는 X_test와 y_test로 진행한다. 즉, X_test와 y_test는 평가에만 사용되고 학습에는 사용되지 않는다.
    history = model.fit_generator(
                    train_generator,
                    steps_per_epoch=100,
                    epochs=100,
                    validation_data=test_generator,
                    validation_steps=100,
                    workers = 10000)
    
    plot_loss_curve(history.history)
    #로스값이 작다고 무조건 좋은것은 아니다. 로스값이 너무 작으면 오버피팅이 일어났다는 증거이다.
    #트레인의 로스가 계속 감소하는데, 밸리데이션의 로스가 증가한다? 오버피팅이니까 중지해라
    print(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])    
    
    model.save('/gdrive/My Drive/Colab/대개론_CNN/googlenet_v4_epochs100_not_rescale.model')
    
    return model
  
def predict_image_sample(model, X_test, y_test, test_id=-1):
    #위에서 받은 test_id가 -1이라면 숫자 10000중에 랜덤으로 하나 뽑음
    #아니면 어떤 값을 받았으면 그 값을 test_sample_id로 저장한다.
    if test_id < 0:
        from random import randrange
        test_sample_id = randrange(10000)
    else:
        test_sample_id = test_id
    
    #X_test에서 테스트샘플아이디에 맞는 이미지를 불러온다.
    test_image = X_test[test_sample_id]
    
    #불러온 이미지를 화면에 보여준다.
    plt.imshow(test_image, cmap='gray')
    
    #차원을 다시 변경해준다. 이거 하나밖에 없으니까 맨 처음 숫자는 1이다 (원래 트레인은 60000)
    test_image = test_image.reshape(1,28,28,1)

    #실제값을 y_actual에 저장
    y_actual = y_test[test_sample_id]
    print("y_actual number=", y_actual)
    
    #예측을 함
    y_pred = model.predict(test_image)
    print("y_pred=", y_pred)
    #y_pred는 소프트맥스 형식으로 확률값들을 보내줌
    #argmax는 최대값을 내보냄.
    y_pred = np.argmax(y_pred, axis=1)[0]
    print("y_pred number=", y_pred)
    
    '''
    만약 예측값과 실제값이 같지 않다면 샘플에 저장함 (안해도 됨)
    if y_pred != y_actual:
        print("sample %d is wrong!" %test_sample_id)
        with open("/gdrive/My Drive/Colab/대개론_CNN/wrong_samples.txt", "a") as errfile:
            print("%d"%test_sample_id, file=errfile)
    else:
        print("sample %d is correct!" %test_sample_id)
    '''    
if __name__ == '__main__': 
   #트레인 모델 실행하면 모델 만들고 저장까지 함
    train_mnist_model()
    #model = load_model('mnist.model')

    #predict_image_sample(model, X_test, y_test)
    
    #for i in range(500):
    #    predict_image_sample(model, X_test, y_test)
