import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:

    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def get_base_model(self):

        self.model = tf.keras.applications.ResNet50(
            input_shape=self.config.params_image_size,
            include_top=self.config.params_include_top,
            weights=self.config.params_weights
        )

        self.save_model(
            path=self.config.base_model_path,
            model=self.model
        )

    @staticmethod
    def _prepare_full_model(
        model,
        classes,
        freeze_all,
        freeze_till,
        learning_rate
    ):

        if freeze_all:

            for layer in model.layers:
                layer.trainable = False

        elif freeze_till is not None and freeze_till > 0:

            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

            for layer in model.layers[-freeze_till:]:
                layer.trainable = True

        gap = tf.keras.layers.GlobalAveragePooling2D()(
            model.output
        )

        dropout = tf.keras.layers.Dropout(
            0.5
        )(gap)

        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(dropout)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        full_model.summary()

        return full_model

    def update_base_model(self):

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,

            freeze_all=False,
            freeze_till=5,

            learning_rate=self.config.params_learning_rate
        )

        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )