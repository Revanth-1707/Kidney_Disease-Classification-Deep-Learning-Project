import os
import math
from pathlib import Path
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detected:", gpus)
    except RuntimeError as e:
        raise e

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path, compile=False)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    def train_valid_generator(self):
        datagenerator_kwargs = dict(rescale=1./255, validation_split=0.20)
        dataflow_kwargs = dict(target_size=self.config.params_image_size[:-1], batch_size=self.config.params_batch_size, interpolation="bilinear")
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)


        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory = self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augumentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.1,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory = self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        print("STEP 1: Starting train()")
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = math.ceil(self.valid_generator.samples // self.valid_generator.batch_size)
        print("STEP 2: Calculated steps")
        callbacks = [
        ModelCheckpoint(
            filepath=self.config.trained_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
        print("STEP 3: Callbacks created")
        history = self.model.fit( 
            self.train_generator, 
            epochs=self.config.params_epochs, 
            steps_per_epoch=self.steps_per_epoch, 
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps, 
            callbacks=callbacks, 
            verbose=1 ) 
        print("STEP 4: model.fit completed")
        return history