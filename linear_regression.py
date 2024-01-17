import numpy as np

class LinearRegression:
    def __init__(self):
        self.parameters = []

    def train(self, training_data, learning_rate):
        # --- DATA PROCESSING ---:
        """
        - The training data should be in the form of a 2D numpy array.
            - The last column should contain the output variable.
            - All columns up to and including the penultimate column should contain input variables.
        """
        training_data_dimensions = training_data.shape
        number_of_training_examples = training_data_dimensions[0]
        number_of_features = training_data_dimensions[1] - 1

        # Extracting input features and output variable as separate numpy arrays:
        input_features = np.delete(training_data, number_of_features, 1)
        output_variable = training_data[:, number_of_features]

        # Adding an extra column to the 2-D array of features.
        # This will represent 'x_0', the dummy parameter, which is set to 1.
        input_features = np.insert(input_features, 0, 1, axis = 1)

        # --- END OF DATA PROCESSING ---

        # --- TRAINING LOOP ---:
        # Initialising all parameters equal to zero.
        self.parameters = [0] * (number_of_features + 1)

        while(not self.model_has_converged()):
            self.update_parameters(learning_rate, input_features, output_variable)

    def hypothesis(self, input_features):
        hypothesis = 0
        for parameter, input in zip(self.parameters, input_features):
            hypothesis += parameter * input
        return hypothesis

    def cost_function(self, input_features, output_variable):
        cost_function = 0
        for row_of_features, output_value in zip(input_features, output_variable):
            cost_function += 0.5 * (self.hypothesis(row_of_features) - output_value) ** 2
        return cost_function

    def model_has_converged(self):
        return False
        # TO BE CONTINUED FROM HERE
        """
        Things to be done:
        - Add A LOT of comments.
        - Think about code design and need for refactoring.
        - Read convergence theory and about when to stop gradient descent.
        - Think about how to test the model.
        """

    def update_parameters(self, learning_rate, input_features, output_variable):
        for parameter, feature in zip(self.parameters, input_features):
            derivative_sum = 0
            for row_of_features, output_value in zip(input_features, output_variable):
                derivative_sum += feature * (self.hypothesis(row_of_features) - output_value)

            parameter = parameter - learning_rate * derivative_sum

# Testing:
regression_model = LinearRegression()
sample_data_list = [[1, 2], [1, 2]]
training_data = np.array(sample_data_list)
# regression_model.parameters = [1, 1]
regression_model.train(training_data)