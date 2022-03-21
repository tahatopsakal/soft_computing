"""
Single Layer Perceptron for linear seperation between 2 datasets
"""

from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


class Perceptron:
    def __init__(self,l_rate,n_iters,random_state = 2 , no_of_samples = 100):

        """
        Initilizer for Perceptron Class

        :param l_rate: Set the learning rate
        :param n_iters: Set the maximum number of iteration
        :param random_state: At each call , a random dataset is generated on fly. Random state of this dataset can be
                             changed using this.
        :param no_of_samples: Set the number of data samples
        """

        self.l_rate = l_rate
        self.n_iters = n_iters
        self.random_state = random_state
        self.no_of_samples = no_of_samples
        self.X , self.y = datasets.make_blobs(n_samples=self.no_of_samples,n_features=2,centers=2,random_state=self.random_state)

    def step_func(self,weighted_sum):
        """
        Using heaviside step function as an activation function.
        :param weighted_sum: transpose(weights)*inputs
        :return: Returns the activated value. 1 if weighted sum is greater than 0 , 0 otherwise
        """
        return 1 if weighted_sum>0 else 0

    def run_perceptron(self):
        """

        Trains the optimal weights

        """
        self.row, self.col = self.X.shape # Number of row and columns in dataset
                                            # Row indicated total number of data
                                            # Col indicated x,y coordinates of the data

        self.weights = np.zeros((self.col + 1, 1)) # Initial weights are selected to be zero

        for _ in range(self.n_iters):
            for idx, data in enumerate(self.X):
                data = np.insert(data, 0, 1).reshape(-1, 1) # Inserting an inital bias value for every input , x0 = 1
                y_hat = self.step_func(np.dot(self.weights.T, data))
                if y_hat - self.y[idx] != 0:
                    self.weights += self.l_rate * ((self.y[idx] - y_hat) * data)

        return self.weights

    def run_and_plot(self):
        """

        Calls run_perceptron method and plots the datasets

        Check the link below for a better understanding

        https://www.thomascountz.com/2018/04/13/
        calculate-decision-boundary-of-perceptron#:~:
        text=A%20perceptron%20is%20more%20specifically,that%20line%20a%20decision%20boundary.

        :return: Returns the dataset and linearly seperated dataset

        """

        final_weights = self.run_perceptron()

        #  Classifier line is  y=mx+c
        #  mx+c = weight0.X0 + weight1.X1 + weight2.X2
        x1 = [min(self.X[:, 0]), max(self.X[:, 0])]
        m = -final_weights[1] / final_weights[2]
        c = -final_weights[0] / final_weights[2]
        x2 = m * x1 + c

        figure1 = plt.figure(figsize=(10, 8))
        plt.plot(self.X[:, 0][self.y == 0], self.X[:, 1][self.y == 0], "r^")
        plt.plot(self.X[:, 0][self.y == 1], self.X[:, 1][self.y == 1], "bs")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title("Perceptron Algorithm for Linear Classification")
        plt.plot(x1, x2, 'r-')
        plt.show()

    @staticmethod
    def get_help():
        """
        Calls the PNG Files
        :return: Returns the PNG Files
        """
        try:
            image_algorithm = Image.open('algorithm.PNG')
            image_perceptron = Image.open('perceptron_idea.PNG')

            image_perceptron.show()
            image_algorithm.show()
        except:
            print('Make sure that the PNG files are located within the same folder with this script')
            return False

        return





perceptron = Perceptron(l_rate=0.2,n_iters=1000,random_state=2 ,no_of_samples=2000)
perceptron.run_and_plot()