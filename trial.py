import numpy as np
import linear_regression_model as lm
import training_data as td

data_list = [[i, 2 * i] for i in range(15)]
print(data_list)
training_data = np.array(data_list)

model = lm.LinearRegressionModel(training_data)
model.train()

print("Model parameters after training:", model.parameters)
print()

testing_values = np.array([1, 1])
print("Testing data provided:", testing_values)
print("Predited value:", model.hypothesis(testing_values))

