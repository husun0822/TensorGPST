import copy, random
import numpy as np
import prox_tv as ptv
import matplotlib.pyplot as plt
from utils import *


# implementation of our Tensor-GPST method
class TensorGPST:

    def __init__(self, param, data):
        self.param = param
        self.N_train, self.N_test = param['N-train'], param['N-test']
        self.H, self.W, self.C = data.X.shape[2], data.X.shape[3], data.X.shape[1]  # dimensionality parameter
        self.h, self.w = param['Latent-Dim'] # latent tensor's dimensionality
        self.r1, self.r2, self.r3 = param['Latent-Rank']  # low-rank approximation parameter
        self.lb = param['lambda']  # regularization tuning parameters
        
        # one can supply a binary mask in the input to impose hard constraint on the sparsity of A and B
        if hasattr(data, 'A'):
            self.A_mask, self.B_mask = (data.A == 0), (data.B == 0)
        else:
            self.A_mask, self.B_mask = None, None
        random.seed(param['seed'])

        # ----- Initialize A & B ----- #
        if self.param['Init-A'] == "random":
            A = np.random.normal(loc=0, scale=1.0, size=self.h * self.H).reshape((self.h, self.H))
            const = np.linalg.norm(A)
            self.A = A / const
        elif self.param['Init-A'] == "fixed":
            self.A = data.A # fixed A based on user input
        elif self.param['Init-A'] == "hard":
            A = np.random.normal(loc=0, scale=1.0, size=self.h * self.H).reshape((self.h, self.H))
            A[self.A_mask] = 0.0 # random initialization based on user input sparsity
            const = np.linalg.norm(A)
            self.A = A / const

        if self.param['Init-B'] == "random":
            self.B = np.random.normal(loc=0, scale=1.0, size=self.w * self.W).reshape((self.w, self.W))
        elif self.param['Init-B'] == "fixed":
            self.B = data.B # fixed B based on user input
        else:
            self.B = np.random.normal(loc=0, scale=1.0, size=self.w * self.W).reshape((self.w, self.W))
            self.B[self.B_mask] = 0.0 # random initialization based on user input sparsity

        # ----- Initialize U1, U2, U3 ----- #
        if self.param['Init-U'] == "random":
            U1 = np.random.normal(loc=0, scale=1.0, size=self.h * self.r1).reshape((self.r1, self.h))
            U2 = np.random.normal(loc=0, scale=1.0, size=self.w * self.r2).reshape((self.r2, self.w))
            U3 = np.random.normal(loc=0, scale=1.0, size=self.C * self.r3).reshape((self.r3, self.C))
            const_U1, const_U2 = np.sqrt(np.linalg.norm(U1.T @ U1)), np.sqrt(np.linalg.norm(U2.T @ U2))
            self.U1, self.U2, self.U3 = U1 / const_U1, U2 / const_U2, U3
        else:
            self.U1 = np.linalg.cholesky(data.K1).T
            self.U2 = np.linalg.cholesky(data.K2).T
            self.U3 = np.linalg.cholesky(data.K3).T

            # ----- Initialize Idiosyncratic Noise ----- #
        self.sigma = 0.5

        # ----- Extract Data ----- #
        self.train_X, self.train_y = data.train_X.transpose((0, 2, 3, 1)).reshape((self.N_train, -1),order="F").transpose(), np.expand_dims(data.train_y, axis=1)
        self.test_X = data.test_X.transpose((0, 2, 3, 1)).reshape((self.N_test, -1), order="F").transpose()

    def Loss(self, A, B, U1, U2, U3, eta):
        # compute the negative log-likehood
        sigma = np.exp(eta / 2)
        U = (np.kron(U3, np.kron(U2, U1)) @ np.kron(np.identity(self.C), np.kron(B, A)) @ self.train_X).T
        K = U @ (U.T) + np.identity(self.N_train) * (sigma ** 2)
        l = np.log(max(np.linalg.det(K), 1e-4)) + self.train_y.T @ np.linalg.inv(K) @ self.train_y
        return l * 0.5

    def partial_LU(self, A, B, U1, U2, U3, sigma):
        U_321 = np.kron(U3, np.kron(U2, U1))
        U = (U_321 @ np.kron(np.identity(self.C), np.kron(B, A)) @ self.train_X).T
        Sigma = np.identity(self.r1 * self.r2 * self.r3) * (sigma ** 2) + U.T @ U
        Sigma_inv = np.linalg.inv(Sigma)
        omega = Sigma_inv @ (U.T) @ self.train_y
        partial_LU = U @ (Sigma_inv + omega @ omega.T / (sigma ** 2)) - (self.train_y @ omega.T) / (sigma ** 2)

        return partial_LU

    def fit(self, max_iter=100, lr=1e-2, tol=1e-4, print_freq=100):
        # ----- set up algorithm iteration tracker ----- #
        delta, Iter = 1, 1
        learning_rate = lr
        delta_hist, loss_hist = [], [],

        # ----- create old/new copy of model parameters ----- #
        A, B, U1, U2, U3 = copy.deepcopy(self.A), copy.deepcopy(self.B), copy.deepcopy(self.U1), copy.deepcopy(
            self.U2), copy.deepcopy(self.U3)
        A_old, B_old, U1_old, U2_old, U3_old = copy.deepcopy(self.A), copy.deepcopy(self.B), copy.deepcopy(
            self.U1), copy.deepcopy(self.U2), copy.deepcopy(self.U3)
        eta, eta_old = 2 * np.log(self.sigma), 2 * np.log(self.sigma)

        # ----- keep track of the best model parameter ----- #
        A_best, B_best, U1_best, U2_best, U3_best, eta_best = copy.deepcopy(self.A), copy.deepcopy(self.B), copy.deepcopy(self.U1), copy.deepcopy(self.U2), copy.deepcopy(self.U3), 2 * np.log(self.sigma)
        nll = self.Loss(A, B, U1, U2, U3, eta)
        tv_penalty = self.lb * (l1_norm(gradient_x(B)) * l1_norm(A) + l1_norm(gradient_x(A)) * l1_norm(B))
        best_loss = nll + tv_penalty


        # ----- Proximal Gradient Descent Algorithm ----- #
        while delta >= tol and Iter <= max_iter:
            # ----- Step for Updating A ----- #
            pLU = self.partial_LU(A_old, B_old, U1_old, U2_old, U3_old, eta_old)
            LU_X = self.train_X @ pLU @ np.kron(U3_old, np.kron(U2_old, U1_old))
            grad_A = np.zeros_like(A)
            for i in range(self.h):
                for j in range(self.H):
                    O_ij = np.zeros((self.h, self.H))
                    O_ij[i, j] = 1.0
                    grad_A[i, j] = np.trace(np.kron(np.identity(self.C), np.kron(B_old, O_ij)) @ LU_X)

            if self.param['Init-A'] == "random":
                A_propose = A_old - learning_rate * grad_A  # propose a gradient descent update
                for i in range(self.h):
                    A_tv = ptv.tv1_1d(A_propose[i, :], w=self.lb * learning_rate * l1_norm(B))  # 1-norm total variation signal approximation
                    A[i, :] = soft_threshold(A_tv,l=self.lb * learning_rate * l1_norm(gradient_x(B)))  # soft-thresholding
            elif self.param['Init-A'] == "hard":
                grad_A[self.A_mask] = 0.0
                A = A - learning_rate * grad_A

            # ----- Step B ----- #
            pLU = self.partial_LU(A, B_old, U1_old, U2_old, U3_old, eta_old)
            LU_X = self.train_X @ pLU @ np.kron(U3_old, np.kron(U2_old, U1_old))
            grad_B = np.zeros_like(B)
            for i in range(self.w):
                for j in range(self.W):
                    O_ij = np.zeros((self.w, self.W))
                    O_ij[i, j] = 1.0
                    grad_B[i, j] = np.trace(np.kron(np.identity(self.C), np.kron(O_ij, A)) @ LU_X)

            if self.param['Init-B'] == "random":
                B_propose = B_old - learning_rate * grad_B  # propose a gradient descent update
                for i in range(self.w):
                    B_tv = ptv.tv1_1d(B_propose[i, :], w=self.lb * learning_rate * l1_norm(A))  # 1-norm total variation signal approximation
                    B[i, :] = soft_threshold(B_tv,l=self.lb * learning_rate * l1_norm(gradient_x(A)))  # soft-thresholding
            elif self.param['Init-B'] == "hard":
                grad_B[self.B_mask] = 0.0
                B = B - learning_rate * grad_B

            # re-normalize the scale of A and B
            const_A = np.linalg.norm(A)
            A = A / (const_A + 1e-8)
            B = B * const_A

            # ----- Step U1 & U2 & U3 ----- #
            pLU = self.partial_LU(A, B, U1_old, U2_old, U3_old, eta_old)
            LU_X = np.kron(np.identity(self.C), np.kron(B, A)) @ self.train_X @ pLU

            # gradient of U1
            grad_U1 = np.zeros_like(U1)
            for i in range(self.r1):
                for j in range(self.h):
                    O_ij = np.zeros((self.r1, self.h))
                    O_ij[i, j] = 1.0
                    grad_U1[i, j] = np.trace(np.kron(U3, np.kron(U2, O_ij)) @ LU_X)

            # gradient of U2
            grad_U2 = np.zeros_like(U2)
            for i in range(self.r2):
                for j in range(self.w):
                    O_ij = np.zeros((self.r2, self.w))
                    O_ij[i, j] = 1.0
                    grad_U2[i, j] = np.trace(np.kron(U3, np.kron(O_ij, U1)) @ LU_X)

            # gradient of U3
            grad_U3 = np.zeros_like(U3)
            for i in range(self.r3):
                for j in range(self.C):
                    O_ij = np.zeros((self.r3, self.C))
                    O_ij[i, j] = 1.0
                    grad_U3[i, j] = np.trace(np.kron(O_ij, np.kron(U2, U1)) @ LU_X)

            # apply the gradient update
            if self.param['Init-U'] == "random":
                U1 = U1_old - learning_rate * grad_U1
                U2 = U2_old - learning_rate * grad_U2
                U3 = U3_old - learning_rate * grad_U3

            # ----- Step sigma ----- #
            U_321 = np.kron(U3, np.kron(U2, U1))
            U = (U_321 @ np.kron(np.identity(self.C), np.kron(B, A)) @ self.train_X).transpose()
            S = np.identity(self.N_train) * np.exp(eta_old) + U @ U.T
            S_inv = np.linalg.inv(S)
            grad_eta = np.trace(np.exp(eta_old) * (0.5 * S_inv - 0.5 * S_inv @ (self.train_y @ self.train_y.T) @ S_inv))

            if self.param['sigma-init'] == "random":
                eta = eta_old - learning_rate * grad_eta

                # ----- track parameter relative change & loss history ----- #
            delta = (np.linalg.norm(A - A_old)) ** 2 + (np.linalg.norm(B - B_old)) ** 2 + (np.linalg.norm(U1 - U1_old)) ** 2 + (np.linalg.norm(U2 - U2_old)) ** 2 + (np.linalg.norm(U3 - U3_old)) ** 2 + (eta - eta_old) ** 2
            delta = np.sqrt(delta)
            nll = self.Loss(A, B, U1, U2, U3, eta)
            tv_penalty = self.lb * (l1_norm(gradient_x(B)) * l1_norm(A) + l1_norm(gradient_x(A)) * l1_norm(B))
            total_loss = nll + tv_penalty
            delta_hist.append(delta)
            loss_hist.append(total_loss)
            self.sigma = np.sqrt(np.exp(eta))

            if total_loss < best_loss:
                best_loss = total_loss
                A_best, B_best, U1_best, U2_best, U3_best, eta_best = copy.deepcopy(A), copy.deepcopy(B), copy.deepcopy(U1), copy.deepcopy(U2), copy.deepcopy(U3), copy.deepcopy(eta)

            if Iter % print_freq == 0:
                print(f"Iter = {Iter}, NLL = {round(nll[0, 0], 4)}, TV-Penalty = {round(tv_penalty, 4)}, Delta = {round(delta, 6)}, sigma = {round(self.sigma, 4)}")

            # overwrite the iterative value
            A_old, B_old, U1_old, U2_old, U3_old, eta_old = copy.deepcopy(A), copy.deepcopy(B), copy.deepcopy(U1), copy.deepcopy(U2), copy.deepcopy(U3), copy.deepcopy(eta)
            Iter += 1
            learning_rate = lr

        # record the estimators
        self.A, self.B, self.U1, self.U2, self.U3, self.sigma = A_best, B_best, U1_best, U2_best, U3_best, np.sqrt(np.exp(eta_best))
        self.delta_hist, self.loss_hist, self.best_loss = delta_hist, loss_hist, best_loss
        # print(f"Algorithm terminated at iteration = {Iter}")

        # make test set prediction
        D_inv = np.identity(self.N_train) / (self.sigma ** 2)
        BA, U321 = np.kron(np.identity(self.C), np.kron(B, A)), np.kron(U3, np.kron(U2, U1))
        U = (U321 @ BA @ self.train_X).transpose()
        Sigma = np.identity(self.r1 * self.r2 * self.r3) * (self.sigma ** 2) + U.T @ U
        KD_inv = D_inv - D_inv @ U @ np.linalg.inv(Sigma) @ U.T
        K_test_train = self.test_X.T @ BA.T @ U321.T @ U321 @ BA @ self.train_X
        K_test_test = self.test_X.T @ BA.T @ U321.T @ U321 @ BA @ self.test_X
        y_pred = K_test_train @ KD_inv @ self.train_y
        y_pred_sigma = K_test_test - K_test_train @ KD_inv @ K_test_train.T
        self.y_pred = y_pred
        self.y_pred_s2 = y_pred_sigma

        # make train set prediction
        K_train_train = self.train_X.T @ BA.T @ U321.T @ U321 @ BA @ self.train_X
        self.y_train_pred = K_train_train @ KD_inv @ self.train_y
        self.y_train_pred_s2 = K_train_train - K_train_train @ KD_inv @ K_train_train

        # make plot for delta and loss history
        if self.param['plot']:
            fig, ax = plt.subplots(1, 2, figsize=(20, 8))

            ax[0].plot(delta_hist)
            ax[0].set_title("Relative Change of Model Estimators")

            ax[1].plot(np.array(loss_hist).squeeze())
            ax[1].set_title("Change of loss history")


# implement the Tensor Gaussian Process with Multi-Linear Kernel
class TensorGP:

    def __init__(self, param, data):
        self.param = param
        self.N_train, self.N_test = param['N-train'], param['N-test']
        self.H, self.W, self.C = data.X.shape[2], data.X.shape[3], data.X.shape[1]  # dimensionality parameter
        self.r1, self.r2, self.r3 = param['Latent-Rank']  # low-rank approximation parameter
        random.seed(param['seed'])

        # ----- Initialize U1, U2, U3 ----- #
        self.U1 = np.random.normal(loc=0, scale=1.0, size=self.H * self.r1).reshape((self.r1, self.H))
        self.U2 = np.random.normal(loc=0, scale=1.0, size=self.W * self.r2).reshape((self.r2, self.W))
        self.U3 = np.random.normal(loc=0, scale=1.0, size=self.C * self.r3).reshape((self.r3, self.C))

        # ----- Initialize Idiosyncratic Noise ----- #
        self.sigma = 0.5

        # ----- Extract Data ----- #
        self.train_X, self.train_y = data.train_X.transpose((0, 2, 3, 1)).reshape((self.N_train, -1),order="F").transpose(), np.expand_dims(data.train_y, axis=1)
        self.test_X = data.test_X.transpose((0, 2, 3, 1)).reshape((self.N_test, -1), order="F").transpose()

    def Loss(self, U1, U2, U3, eta):
        # compute the negative log-likehood
        sigma = np.exp(eta / 2)
        U = (np.kron(U3, np.kron(U2, U1)) @ self.train_X).T
        K = U @ (U.T) + np.identity(self.N_train) * (sigma ** 2)
        l = np.log(max(np.linalg.det(K), 1e-4)) + self.train_y.T @ np.linalg.inv(K) @ self.train_y
        return l * 0.5

    def partial_LU(self, U1, U2, U3, sigma):
        U_321 = np.kron(U3, np.kron(U2, U1))
        U = (U_321 @ self.train_X).T
        Sigma = np.identity(self.r1 * self.r2 * self.r3) * (sigma ** 2) + U.T @ U
        Sigma_inv = np.linalg.inv(Sigma)
        omega = Sigma_inv @ (U.T) @ self.train_y
        partial_LU = U @ (Sigma_inv + omega @ omega.T / (sigma ** 2)) - (self.train_y @ omega.T) / (sigma ** 2)

        return partial_LU

    def fit(self, max_iter=100, lr=1e-2, tol=1e-4, print_freq=100):
        # ----- set up algorithm iteration tracker ----- #
        delta, Iter = 1, 1
        learning_rate = lr
        delta_hist, loss_hist = [], []

        # ----- create old/new copy of model parameters ----- #
        U1, U2, U3 = copy.deepcopy(self.U1), copy.deepcopy(self.U2), copy.deepcopy(self.U3)
        U1_old, U2_old, U3_old = copy.deepcopy(self.U1), copy.deepcopy(self.U2), copy.deepcopy(self.U3)
        eta, eta_old = 2 * np.log(self.sigma), 2 * np.log(self.sigma)

        # ----- Gradient Descent Algorithm ----- #
        while delta >= tol and Iter <= max_iter:
            # ----- Step U1 & U2 & U3 ----- #
            pLU = self.partial_LU(U1_old, U2_old, U3_old, eta_old)
            LU_X = self.train_X @ pLU

            # gradient of U1
            grad_U1 = np.zeros_like(U1)
            for i in range(self.r1):
                for j in range(self.H):
                    O_ij = np.zeros((self.r1, self.H))
                    O_ij[i, j] = 1.0
                    grad_U1[i, j] = np.trace(np.kron(U3, np.kron(U2, O_ij)) @ LU_X)

            # gradient of U2
            grad_U2 = np.zeros_like(U2)
            for i in range(self.r2):
                for j in range(self.W):
                    O_ij = np.zeros((self.r2, self.W))
                    O_ij[i, j] = 1.0
                    grad_U2[i, j] = np.trace(np.kron(U3, np.kron(O_ij, U1)) @ LU_X)

            # gradient of U3
            grad_U3 = np.zeros_like(U3)
            for i in range(self.r3):
                for j in range(self.C):
                    O_ij = np.zeros((self.r3, self.C))
                    O_ij[i, j] = 1.0
                    grad_U3[i, j] = np.trace(np.kron(O_ij, np.kron(U2, U1)) @ LU_X)

            # apply the gradient update
            U1 = U1_old - learning_rate * grad_U1
            U2 = U2_old - learning_rate * grad_U2
            U3 = U3_old - learning_rate * grad_U3

            # ----- Step sigma ----- #
            U_321 = np.kron(U3, np.kron(U2, U1))
            U = (U_321 @ self.train_X).transpose()
            S = np.identity(self.N_train) * np.exp(eta_old) + U @ U.T
            S_inv = np.linalg.inv(S)
            grad_eta = np.trace(np.exp(eta_old) * (0.5 * S_inv - 0.5 * S_inv @ (self.train_y @ self.train_y.T) @ S_inv))
            eta = eta_old - learning_rate * grad_eta

            # ----- track parameter relative change & loss history ----- #
            delta = (np.linalg.norm(U1 - U1_old)) ** 2 + (np.linalg.norm(U2 - U2_old)) ** 2 + (np.linalg.norm(U3 - U3_old)) ** 2 + (eta - eta_old) ** 2
            delta = np.sqrt(delta)
            nll = self.Loss(U1, U2, U3, eta)
            delta_hist.append(delta)
            loss_hist.append(nll)
            self.sigma = np.sqrt(np.exp(eta))

            if Iter % print_freq == 0:
                print(f"Iter = {Iter}, Loss = {round(nll[0, 0], 4)}, Delta = {round(delta, 6)}, sigma = {round(self.sigma, 4)}")

            # overwrite the iterative value
            U1_old, U2_old, U3_old, eta_old = copy.deepcopy(U1), copy.deepcopy(U2), copy.deepcopy(U3), copy.deepcopy(eta)
            Iter += 1
            learning_rate = lr

        # record the estimators
        self.U1, self.U2, self.U3, self.sigma = U1, U2, U3, np.sqrt(np.exp(eta))
        self.delta_hist, self.loss_hist = delta_hist, loss_hist
        # print(f"Algorithm terminated at iteration = {Iter}")

        # make test set prediction
        D_inv = np.identity(self.N_train) / (self.sigma ** 2)
        U321 = np.kron(U3, np.kron(U2, U1))
        U = (U321 @ self.train_X).transpose()
        Sigma = np.identity(self.r1 * self.r2 * self.r3) * (self.sigma ** 2) + U.T @ U
        KD_inv = D_inv - D_inv @ U @ np.linalg.inv(Sigma) @ U.T
        K_test_train = self.test_X.T @ U321.T @ U321 @ self.train_X
        K_test_test = self.test_X.T @ U321.T @ U321 @ self.test_X
        y_pred = K_test_train @ KD_inv @ self.train_y
        y_pred_sigma = K_test_test - K_test_train @ KD_inv @ K_test_train.T
        self.y_pred = y_pred
        self.y_pred_s2 = y_pred_sigma

        # make train set prediction
        K_train_train = self.train_X.T @ U321.T @ U321 @ self.train_X
        self.y_train_pred = K_train_train @ KD_inv @ self.train_y
        self.y_train_pred_s2 = K_train_train - K_train_train @ KD_inv @ K_train_train

        # make plot for delta and loss history
        if self.param['plot']:
            fig, ax = plt.subplots(1, 2, figsize=(20, 8))

            ax[0].plot(delta_hist)
            ax[0].set_title("Relative Change of Model Estimators")

            ax[1].plot(np.array(loss_hist).squeeze())
            ax[1].set_title("Change of loss history")