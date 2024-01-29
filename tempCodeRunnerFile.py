import numpy as np
import linear_regression_model as lm

data_list = [[1, 1, 1, 2],
             [1, 2, 1, 3],
             [1, 1, 3, 4]]
training_data = np.array(data_list)

model = lm.LinearRegressionModel(training_data)
model.train()
print(model.cost_function_history)
print(model.parameters)