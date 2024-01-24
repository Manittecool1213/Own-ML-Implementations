"""
    Things to be done:
    - Add A LOT of comments.
    - Think about code design and need for refactoring.
    - Read convergence theory and about when to stop gradient descent. DONE
    - Think about how to test the model.
"""

import numpy as np

class LinearRegressionModel:
    def __init__(self):
        # Fundamental constants used in the model:
        self.RELATIVE_COST_CHANGE_THRESHOLD = 0.001
        """
        - The relative change in cost function between successive iterations should fall below this value for convergence to be achieved.
        - It is meant to represent a relative change. Percentage values need to be converted into decimals.
        - e.g] a treshold of 10% should be represented as 0.1
        """

        self.MAX_ALLOWED_ITERATIONS = 1000
        """
        -
        """

        self.parameters = []
        self.cost_function_history = [0]
        """
        - The initial cost value is set to zero in order to allow the firs comparison.
        - If the list was left blank, there would be no value to check against to test for convergence.
        """
        self.iteration_count = 0

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
        while(not self.model_has_converged(input_features, output_variable)):
            self.iteration_count += 1
            self.update_parameters(learning_rate, input_features, output_variable)

    def hypothesis(self, input_features):
        hypothesis = 0
        for parameter, input in zip(self.parameters, input_features):
            hypothesis += parameter * input
        return hypothesis

    def cost_function(self, input_features, output_variable):
        cost = 0
        for row_of_features, output_value in zip(input_features, output_variable):
            cost += 0.5 * (self.hypothesis(row_of_features) - output_value) ** 2
        return cost

    def model_has_converged(self, input_features, output_variable):
        """
        Approach used to check for convergence:
        - The approach combines two concepts: relative change in cost function and an iteration limit.
        - Cost function:
            - As the model keeps training, the cost function should decrease.
            - As the model approaches its final form, the changes in the cost function should keep getting smaller.
            - Thus, convergence is tested for by checking whether the relative change in the value of the cost has fallen below a certain pre-determined threshold.
        - Iteration limit:
            - In some cases, training may be too slow and the change in the cost may not satisfy the desired conditions.
            - To avoid extremely long training times, a hard iteration limit is imposed.
        """
        if self.iteration_count > self.MAX_ALLOWED_ITERATIONS:
            return True
        else:
            self.cost_function_history.append(self.cost_function(input_features, output_variable))

            new_cost = self.cost_function_history[-1]
            old_cost = self.cost_function_history[-2]
            relative_change_in_cost = abs((new_cost - old_cost)) / old_cost

            return relative_change_in_cost < self.RELATIVE_COST_CHANGE_THRESHOLD

    def update_parameters(self, learning_rate, input_features, output_variable):
        for parameter, feature in zip(self.parameters, input_features):
            derivative_sum = 0
            for row_of_features, output_value in zip(input_features, output_variable):
                derivative_sum += feature * (self.hypothesis(row_of_features) - output_value)

            parameter = parameter - learning_rate * derivative_sum