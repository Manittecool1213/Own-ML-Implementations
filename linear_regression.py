import numpy as np

class LinearRegression:
    def __init__(self):
        self.parameters = []

    def train(self, training_data):
        """
        - The training data should be in the form of a 2D numpy array.
            - The last column should contain the output variable.
            - All columns up to and including the penultimate column should contain input variables.
        """
        training_data_dimensions = training_data.shape
        number_of_training_examples = training_data_dimensions[0]
        number_of_features = training_data_dimensions[1] - 1

        # Initialising all parameters equal to zero.
        self.parameters = [0] * (number_of_features + 1)

    def hypothesis():
        pass