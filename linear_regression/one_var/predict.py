import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import train

dataset = pd.read_csv("/home/singh/PycharmProjects/MachineLearning_from_Scratch/Datasets/linear_regression"+
                      "/one_var/train.csv")

print("Shape of dataset: " + str(dataset.shape))

data_x = dataset['x'][:50]
data_y = dataset['y'][:50]

data_x = data_x.to_numpy()
data_y = data_y.to_numpy()
data_x = data_x.reshape(len(data_x), 1)
data_y = data_y.reshape(len(data_y), 1)

train.fit()


# hypothesis function of the form y = c1 + c2*x
def h(c1, c2, xh):
    size = xh.shape[0]
    a1 = np.ones((size, 1), dtype=np.float)
    a1 = np.concatenate((a1, xh), axis=1)
    a2 = np.array(([c1],
                  [c2]))
    yh = a1.dot(a2)

    return yh


def plot_all(para1, para2, x_data, y_data):
    y_pred = h(para1, para2, x_data)

    plt.style.use("seaborn")
    fig, (ax_h, ax_train) = plt.subplots(2, 1, sharex=True)

    # plotting the scatter plot of training data
    ax_train.set_title("training data")
    ax_train.set_ylabel("random nums generated")
    ax_train.set_xlabel("data x")
    ax_train.scatter(x_data, y_data)

    # plotting the hypothesis funtion
    ax_h.set_title("h(x) plot")
    ax_h.set_ylabel("h(x)-predictions")
    ax_h.plot(x_data, y_pred)

    plt.show()


plot_all(0, 1, data_x, data_y)
train.plot_j(data_x, data_y)




