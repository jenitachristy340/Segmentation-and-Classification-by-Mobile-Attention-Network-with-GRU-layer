import cv2 as cv
import numpy as np
import tensorflow
from PIL import Image
from keras.applications import MobileNet
# from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
# from tensorflow.python.keras.applications.mobilenet import MobileNet
from keras import activations
import tensorflow as tf

from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from keras.optimizers import RMSprop, Adadelta, Adagrad
from keras.preprocessing import image

from Evaluation import evaluation


def Model_MobileNet(train_data, train_tar, test_data, test_tar, Optimizer, sol='none'):
    if sol is 'none':
        sol = [1, 100, 50]
    model = MobileNet(weights='imagenet')
    IMG_SIZE = [224, 224, 3]
    Feat = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    test_data = Feat.reshape(Feat.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    for i in range(test_data.shape[0]):
        data = Image.fromarray(np.uint8(test_data[i])).convert('RGB')
        data = image.img_to_array(data)
        data = np.expand_dims(data, axis=0)
        data = np.squeeze(data)
        test_data[i] = cv.resize(data, (224, 224))
        test_data[i] = preprocess_input(test_data[i])


    # optimizer = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad']
    model.compile(optimizer=Optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])  # optimizer='Adam'
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy
    model.fit_generator(generator=train_data,
                        steps_per_epoch=int(sol[1]),
                        epochs=int(sol[2]))
    preds = model.predict(test_tar)
    activation = activations.relu(test_data).numpy()
    Eval = evaluation(preds, test_tar)
    pred = decode_predictions(preds, top=3)[0]
    return Eval, pred


def Model_MobileNet_1(train_data, train_tar, test_data, test_tar, Act, sol='none'):
    if sol is 'none':
        sol = [1, 100, 50]
    model = MobileNet(weights='imagenet')
    IMG_SIZE = [224, 224, 3]
    Feat = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    test_data = Feat.reshape(Feat.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    for i in range(test_data.shape[0]):
        data = Image.fromarray(np.uint8(test_data[i])).convert('RGB')
        data = image.img_to_array(data)
        data = np.expand_dims(data, axis=0)
        data = np.squeeze(data)
        test_data[i] = cv.resize(data, (224, 224))
        test_data[i] = preprocess_input(test_data[i])


    optimizer = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad']
    model.compile(optimizer=optimizer[int(sol[0])], loss='categorical_crossentropy',
                  metrics=['accuracy'])  # optimizer='Adam'
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy
    model.fit_generator(generator=train_data,
                        steps_per_epoch=int(sol[1]),
                        epochs=int(sol[2]))
    preds = model.predict(test_tar)
    activation = activations.relu(test_data).numpy()
    Eval = evaluation(preds, test_tar)
    pred = decode_predictions(preds, top=3)[0]
    return Eval, pred
