import numpy as np

"""
- The training data should be in the form of a 2D numpy array.
- The last column should contain the output variable.
- All columns up to and including the penultimate column should contain input variables.
"""

class trainingData:
    def __init__(self, data):
        self.data = data

        # Extracting dimensional information:
        self.dimensions = data.shape
        self.number_of_training_examples = self.dimensions[0]
        self.number_of_features = self.dimensions[1] - 1

        # Extracting input features and output variable from data as separate numpy arrays:
        self.input_features = np.delete(self.data, self.number_of_features, 1)
        # Deleting the last column of the array, which stores the values of the output variable.

        self.input_features = np.insert(self.input_features, 0, 1, axis = 1)
        """
        - Adding an extra column to the 2-D array of features.
        - This will represent 'x_0', the dummy parameter, which is set to 1.
        """

        self.output_variable = self.data[:, self.number_of_features]
        # Selecting only the last column of the array, which stores the values of the output variable.