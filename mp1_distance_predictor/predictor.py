# code reuse partial of https://github.com/harshilpatel312/KITTI-distance-estimation
"""This file contains the NN-based distance predictor.

Here, you will design the NN module for distance prediction
"""
from mp1_distance_predictor.inference_distance import infer_dist
from mp1_distance_predictor.detect import detect_cars

from pathlib import Path
from keras.models import load_model
from keras.models import model_from_json


# NOTE: Very important that the class name remains the same
class Predictor:
    def __init__(self, model_file: Path):
        # TODO: You can use this path to load your trained model.
        self.model_file = model_file
        self.detect_model = None
        self.distance_model = None

    def initialize(self):

        self.detect_model = load_model('mp1_distance_predictor/model.h5')
        self.distance_model = self.load_inference_model()

    def load_inference_model(self):
        MODEL = 'model@1535470106'
        WEIGHTS = 'model@1535470106'

        # load json and create model
        json_file = open('mp1_distance_predictor/distance_model_weights/{}.json'.format(MODEL), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights("mp1_distance_predictor/distance_model_weights/{}.h5".format(WEIGHTS))
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
        return loaded_model


    def predict(self, obs) -> float:
        """This is the main predict step of the NN.

        Here, the provided observation is an Image. Your goal is to train a NN that can
        use this image to predict distance to the lead car.

        """
        data = obs
        image_name = 'camera_images/vision_input.png'
        # load a trained yolov3 model

        car_bounding_box = detect_cars(self.detect_model, image_name)  # return the bounding box of car
        dist_test = data.distance_to_lead
        # Different dist_test will have effect on the prediction
        # You can play with the number of dist_test
        if car_bounding_box is not None:
            dist = infer_dist(self.distance_model, car_bounding_box, [[dist_test]])
        else:
            print("No car detected")
            # If no car detected what would you do for the distance prediction
            # Do your magic...


            dist = 30

        print("estimated distance: ", dist)



        return dist
