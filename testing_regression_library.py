import numpy as np
import linear_regression_model as lm

data_list = [[i, 2 * i] for i in range(20)]
print(data_list)
training_data = np.array(data_list, dtype = np.float64)

model = lm.LinearRegressionModel(training_data)
model.train()

for i in range(100):
    print(f"{i}th cost function: {model.cost_function_history[i]}")

# print("Model parameters after training:", model.parameters)
# print()

# testing_values = np.array([1])
# print("Testing data provided:", testing_values)
# print("Predicted value:", model.hypothesis(testing_values, testing = True))

