import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:

    model = load_model(os.path.join("artifacts", "training", "model.keras"))
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        print("Loading model...")
        
        imagename = self.filename

        print("Model loaded")
        test_image = image.load_img(imagename, target_size = (224, 224))

        print("Image loaded")
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        print("Running prediction...")
        result = np.argmax(PredictionPipeline.model.predict(test_image), axis=1)
        print(result)

        if result[0] == 3:
            prediction = 'Tumor'
            return [{"image" : prediction}]
        
        if result[0] == 2:
            prediction = 'Stone'
            return [{"image" : prediction}]
        
        if result[0] == 1:
            prediction = 'Normal'
            return [{"image" : prediction}]
        
        if result[0] == 0:
            prediction = 'Cyst'
            return [{"image" : prediction}]
