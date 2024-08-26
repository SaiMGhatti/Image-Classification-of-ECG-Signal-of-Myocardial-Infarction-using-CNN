import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Conv1D, MaxPooling2D, MaxPooling1D, Dropout, Flatten, Dense
import keras
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

def input_values(lead):                                      # Grouping the filepaths and labels
    dir = Path('drive/MyDrive/ptb-ecg-db/I')

    filepaths = list(dir.glob(r'**/*.jpeg'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))


    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    dtf3 = pd.concat([filepaths , labels] , axis=1)
    return filepaths,labels
    
def train_test_split(lead):                                  # Splitting training and test dataset based on stratified K-fold.
    filepaths,labels = input_values(lead)
    num_folds = 10  # You can specify the number of folds here
    stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    X = filepaths
    y = labels
    for train_index, test_index in stratified_kfold.split(filepaths, labels):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    dataframe_train = pd.concat([X_train , y_train] , axis=1)
    dataframe_test = pd.concat([X_test , y_test] , axis=1)
    return dataframe_train, dataframe_test


class Classification:
    def __init__(self,lead):
        dataframe_train,dataframe_test = train_test_split(lead)
        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                          preprocessing_function=self.gray_torgb,
                          rescale=1./255,
                          validation_split=0.2)

        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                         preprocessing_function=self.gray_torgb,
                         rescale=1./255)
        size=224
        color_mode='rgb'
        self.batch_size=64

        self.train_images = train_generator.flow_from_dataframe(
            dataframe=dataframe_train,
            x_col='Filepath',
            y_col='Label',
            target_size=(size, size),
            color_mode=color_mode,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True,
            seed=42,
            subset='training'
        )

        self.val_images = train_generator.flow_from_dataframe(
            dataframe=dataframe_train,
            x_col='Filepath',
            y_col='Label',
            target_size=(size, size),
            color_mode=color_mode,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True,
            seed=42,
            subset='validation'
        )

        self.test_images = test_generator.flow_from_dataframe(
            dataframe=dataframe_test,
            x_col='Filepath',
            y_col='Label',
            target_size=(size, size),
            color_mode=color_mode,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.device = "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0"
    
    
    
    def gray_torgb(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.merge((image,image,image))
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image
    
    
    def model(self):
        model = Sequential()

        model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(units=11, activation='softmax'))

        model.summary()
        
        return model
    
    def train_model(self):                                  #Training the model for 10 epochs
        with tf.device(self.device):
            checkpoint = keras.callbacks.ModelCheckpoint(
                            filepath='ptb-ecg-db/best_model.h5',
                            save_weights_only=False,
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True,
                            verbose=1)
                            
            model = self.model()
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',
                         'Precision',
                         'Recall',
                         tf.keras.metrics.AUC(
                                        num_thresholds=200,
                                        curve="ROC",
                                        summation_method="interpolation",
                                        multi_label=False
                                        )])
            steps_per_epoch = len(self.train_images) // batch_size
            validation_steps = len(self.val_images) // batch_size
            
            result=model.fit(
                self.train_images,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.val_images,
                validation_steps=validation_steps,
                epochs=10,
                callbacks=[checkpoint])
            
            print(result[1]*100)
            
            
    def test_model(self):                                    #Testing the model for performance metrics
        best_model=load_model('ptb-ecg-db/best_model.h5')
        results = best_model.evaluate(self.test_images, verbose=0)
        print("     Test Loss: {:.4f}".format(results[0]))
        print(" Test Accuracy: {:.4f}%".format(results[1] * 100))
        print("Test Precision: {:.4f}%".format(results[2] * 100))
        print("   Test Recall: {:.4f}%".format(results[3] * 100))
        y_pred = best_model.predict(self.test_images)
        y_pred = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(self.test_images.labels, y_pred)
        print(cm)

        report = classification_report(self.test_images.labels, y_pred, target_names=['A', 'AL', 'AS', 'H', 'I', 'IL','IP','IPL','L','P','PL'],digits=4)
        print(report)
        
        
c = Classification('I')
c.train_model()
#c.test_model()

        
    
    







