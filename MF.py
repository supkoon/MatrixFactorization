import numpy as np
import pandas as pd
from functions.dataloader import dataloader
import argparse
import matplotlib.pyplot as plt

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="MatrixFactorization-MovieLens.")
    parser.add_argument('--path', nargs='?', default='./datasets/movielens/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ratings.csv',
                        help='Choose a dataset.')
    parser.add_argument('--num_factors', type=int, default=20,
                        help='number of latent_factor MF model.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--MF_regs', type=float, default=0.01,
                        help='Regularization for MF embeddings.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--test_size', type=float,default=0.2,
                        help="test_proportion of datasets")
    return parser.parse_args()


class MatrixFactorization():
    def __init__(self, ratings, test_ratings, latent_factors, learning_rate, confidence, reg, epochs):
        self.ratings = ratings
        self.test_ratings = test_ratings
        self.num_user, self.num_item = ratings.shape
        self.latent_factors = latent_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.confidence = confidence

    def fit(self):

        # latent factors
        self.p_u = np.random.normal(size=(self.num_user, self.latent_factors))
        self.q_i = np.random.normal(size=(self.num_item, self.latent_factors))

        # biases
        self.b_u = np.zeros(self.num_user)
        self.b_i = np.zeros(self.num_item)
        self.b = np.mean(self.ratings[self.ratings != 0])

        self.training_process = []
        for epoch in range(self.epochs):

            for i in range(self.num_user):
                for j in range(self.num_item):
                    if self.ratings[i, j] > 0:
                        self.gradient_descent(i, j, self.ratings[i, j])
            loss = self.loss()
            test_loss = self.test_loss()
            self.training_process.append((epoch, loss, test_loss))

            print("Epoch: %d ; loss = %.4f ; test_loss = %.4f" % (epoch + 1, loss, test_loss))

    def test_loss(self):
        xi, yi = self.test_ratings.nonzero()
        predicted = self.get_whole_prediction()
        test_loss = 0
        for x, y in zip(xi, yi):
            test_loss += np.power(self.test_ratings[x, y] - predicted[x, y], 2)
        return np.sqrt(test_loss) / len(xi)

    def loss(self):
        xi, yi = self.ratings.nonzero()
        predicted = self.get_whole_prediction()
        loss = 0
        for x, y in zip(xi, yi):
            loss += np.power(self.ratings[x, y] - predicted[x, y], 2)
        return np.sqrt(loss) / len(xi)

    def gradient_descent(self, i, j, rating):

        prediction = self.get_each_prediction(i, j)

        dbu = -2 * self.confidence[i][j] * (rating - self.b_u[i] - self.b_i[j] - self.b - prediction) + 2 * self.reg * \
              self.b_u[i]
        dbi = -2 * self.confidence[i][j] * (rating - self.b_u[i] - self.b_i[j] - self.b - prediction) + 2 * self.reg * \
              self.b_i[i]
        self.b_u[i] -= self.learning_rate * dbu
        self.b_i[j] -= self.learning_rate * dbi

        dp = -2 * self.confidence[i][j] * (rating - self.b_u[i] - self.b_i[j] - self.b - prediction) * self.q_i[j,
                                                                                                       :] + 2 * (
                         self.reg * self.p_u[i, :])
        dq = -2 * self.confidence[i][j] * (rating - self.b_u[i] - self.b_i[j] - self.b - prediction) * self.p_u[i,
                                                                                                       :] + 2 * (
                         self.reg * self.q_i[j, :])

        self.p_u[i, :] -= self.learning_rate * dp
        self.q_i[j, :] -= self.learning_rate * dq

    def get_each_prediction(self, i, j):
        return self.b + self.b_u[i] + self.b_i[j] + np.dot(self.p_u[i, :], self.q_i[j, :].T)

    def get_whole_prediction(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + np.dot(self.p_u, self.q_i.T)

    def plot_results(self):
        df = pd.DataFrame(self.training_process)
        df.columns = ["epoch", "train_loss", "test_loss"]
        print(df)
        df.loc[:][["train_loss", "test_loss"]].plot()
        plt.xlabel("epoch")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show()

if __name__=="__main__":
    # argparse
    args = parse_args()
    num_factors = args.num_factors
    regs = args.MF_regs
    learning_rate = args.lr
    epochs = args.epochs
    test_size = args.test_size


    dataloader = dataloader(args.path +args.dataset,test_size)
    train_data = dataloader.get_train_data()
    test_data = dataloader.get_test_data()
    confidence = dataloader.get_confidence()

    MF = MatrixFactorization(train_data,test_data,num_factors,learning_rate,confidence,regs,epochs)
    MF.fit()
    MF.plot_results()


