import pandas as pd
import matplotlib.pyplot as plt


# hypothesis function of the form y = c1 + c2*x
def h(c1, c2, xh):
    yh = c1 + c2*xh
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

data_x = dataset['x']
data_y = dataset['y']

plt.scatter(x, y)
plt.show()


