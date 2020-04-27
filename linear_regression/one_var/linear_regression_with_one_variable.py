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
h_y = h(0, 1, data_x)

plt.style.use("seaborn")
fig, (ax_train, ax_h) = plt.subplots(2, 1, sharex=True)

ax_train.set_title("Training Data Plot")
ax_train.scatter(data_x, data_y)
ax_train.set_ylabel("Random nums generated")

ax_h.set_title("Hypothesis Function Plot")
ax_h.set_xlabel("x_numbers")
ax_h.set_ylabel("predictions by h(x)")
ax_h.plot(data_x, h_y, 'm')

plt.show()




