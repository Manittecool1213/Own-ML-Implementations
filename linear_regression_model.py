"""
    Things to be done:
    - [x] Resolve issue where user has to provide value of x_0 while testing.
    - [ ] Solve issue of gradient descent not working - cost function just keeps increasing for large data.
    - [ ] Think about how to test the model with better datasets.
    - [ ] Play with fundamental constant values to gauge impact on accuracy.
    - [ ] Make notes for convergence theory and stopping gradient descent.
    - [ ] Read about and implement accuracy metrics.
    - [ ] Read about other possible initial parameter values.
    - [ ] System to graph change in every parameter as rounds progress.
"""

import training_data as td

class LinearRegressionModel:
    def __init__(self, training_data, relative_cost_change_threshold = 0.001, max_allowed_iterations = 1000, learning_rate = 0.001):
        # Fundamental constants used in the model:
        self.RELATIVE_COST_CHANGE_THRESHOLD = relative_cost_change_threshold
        """
        - The relative change in cost function between successive iterations should fall below this value for convergence to be achieved.
        - It is meant to represent a relative change. Percentage values need to be converted into decimals.
        - e.g] a threshold of 10% should be represented as 0.1
        """

        self.MAX_ALLOWED_ITERATIONS = max_allowed_iterations
        """
        - In order to save computation time, a hard limit is imposed on the maximum number of training iterations allowed.
        - Irrespective of whether the specified convergence criteria have been achieved, training will stop after this limit is reached.
        """

        self.LEARNING_RATE = learning_rate

        # Additional variables related to the model:
        self.training_data = td.trainingData(training_data)
        """
        - The training data is stored as an object.
        - The trainingData class performs the necessary data formatting and provides the required values as parameters of the data object.
        """

        self.parameters = []
        self.cost_function_history = [1]
        """
        - The initial cost value is set to one in order to allow the first comparison.
        - If the list was left blank, there would be no value to check against to test for convergence.
        - The value is set to 1 instead of zero because the alternative would lead to division by zero.
        """

        self.iteration_count = 0

    def train(self):
        # Initialising all parameters equal to zero:
        self.parameters = [0] * (self.training_data.number_of_features + 1)

        # Update parameters till convergence is achieved or iteration limit is hit:
        while not self.model_has_converged():
            self.iteration_count += 1
            self.update_parameters()

    def model_has_converged(self):
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
            self.cost_function_history.append(self.cost_function())

            new_cost = self.cost_function_history[-1]
            old_cost = self.cost_function_history[-2]
            relative_change_in_cost = abs((new_cost - old_cost)) / old_cost

            return relative_change_in_cost < self.RELATIVE_COST_CHANGE_THRESHOLD

    def update_parameters(self):
        # Updating parameters based on gradient descent algorithm.
        for loop_counter in range(len(self.parameters)):
            derivative_sum = 0
            for row_of_features, output_value in zip(self.training_data.input_features, self.training_data.output_variable):
                derivative_sum += (self.hypothesis(row_of_features) - output_value) * row_of_features[loop_counter]

            self.parameters[loop_counter] = self.parameters[loop_counter] - self.LEARNING_RATE * derivative_sum

    def cost_function(self):
        # The cost function quantifies the difference between the values predicted by the model and the true output variable.
        cost = 0
        for row_of_features, output_value in zip(self.training_data.input_features, self.training_data.output_variable):
            cost += 0.5 * (self.hypothesis(row_of_features) - output_value) ** 2
            # print(f"Cost after newest update: {cost}")
        return cost

    def hypothesis(self, row_of_features, testing = False):
        """
        - The hypothesis function is the prediction which the model makes.
        - It is used in the training as well as testing phases.
        """

        if(testing == True):
            row_of_features = list(row_of_features)
            row_of_features.insert(0, 1)
        """
        - The testing data should be in the form of a 1D numpy array containing the input features.
        - When the function is used for testing, '1' is appended to the inputted row of features.
        - This allows the user to only enter the input features, without the additional '1' corresponding to x_0.
        """

        hypothesis = 0
        for parameter, row_element in zip(self.parameters, row_of_features):
            hypothesis += parameter * row_element
        return hypothesis

    def test_accuracy(self):
        pass