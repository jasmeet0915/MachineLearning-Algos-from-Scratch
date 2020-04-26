import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# hypothesis function of the form y = c1 + c2*x
def h(c1, c2, xh):
    size = xh.shape[0]
    a1 = np.ones((size, 1), dtype=np.float)
    a1 = np.concatenate((a1, xh), axis=1)
    a2 = np.array(([c1],
                  [c2]))
    yh = a1.dot(a2)

    return yh



# cost function of the form of squared mean error function
def j(p1, p2, m, x, y):
    sum = 0

    # loop to calculate the sum of squared errors
    for i in range(0, m):
        sum = sum + (h(p1, p2, x[i]) - y[i])**2

    # calculating the mean of squared errors
    sq_mean_err = sum/m
    sq_mean_err = sq_mean_err/2
    return sq_mean_err


dataset = pd.read_csv("/home/singh/PycharmProjects/MachineLearning_from_Scratch/Datasets/linear_regression"+
                      "/one_var/train.csv")

print("Shape of dataset: " + str(dataset.shape))
print("\ndataset head: \n" + str(dataset.head()))

data_x = dataset['x'][:50]
data_y = dataset['y'][:50]

data_x = data_x.to_numpy()
data_x = data_x.reshape(len(data_x), 1)
y = h(0, 1, data_x)


plt.scatter(data_x, data_y)
plt.plot(data_x, y)
plt.title("Training Data")
plt.xlabel("x value")
plt.ylabel("random y value generated")
plt.show()


