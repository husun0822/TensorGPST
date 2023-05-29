import random, types, math
import numpy as np

# ----- Generate Simulation Data ----- #
def Simulation(param, seed=2023):
    np.random.seed(seed)  # set random seed

    N, train_ratio = param['N'], param['train_ratio']
    H, W, C, h, w = 25, 25, 3, 3, 3
    X, X_channel = [], []

    # generate observable image data (i.e. X)
    for i in range(N):
        # randomly choose a channel to contain signal
        channel = np.random.choice(range(C))
        X_channel.append(channel)
        tensor = np.random.normal(loc=0.0, scale=0.3, size=(H, W, C))

        if channel == 1:
            tensor[10:15, 10:15, 1] = np.random.normal(loc=4.0, scale=0.3, size=(5, 5))
        else:
            location = random.choice(["Top-Left", "Top-Right", "Low-Left", "Low-Right"])
            if location == "Top-Left":
                tensor[2:7, 2:7, channel] = np.random.normal(loc=4.0, scale=0.3, size=(5, 5))
            elif location == "Top-Right":
                tensor[2:7, 18:23, channel] = np.random.normal(loc=4.0, scale=0.3, size=(5, 5))
            elif location == "Low-Left":
                tensor[18:23, 2:7, channel] = np.random.normal(loc=4.0, scale=0.3, size=(5, 5))
            else:
                tensor[18:23, 18:23, channel] = np.random.normal(loc=4.0, scale=0.3, size=(5, 5))

        X.append(tensor)

    X, X_channel = np.array(X).transpose((0, 3, 1, 2)), np.array(X_channel)

    # generate latent image data (i.e. Z)
    A, B = np.zeros((h, H)), np.zeros((w, W))
    A[0, 2:7] = 1/5
    A[1, 10:15] = 1/5
    A[2, 18:23] = 1/5
    B[0, 2:7] = 1/5
    B[1, 10:15] = 1/5
    B[2, 18:23] = 1/5
    Z = A @ X @ B.T

    # generate regression label (i.e. y)
    K1 = np.array([[1.0, 0.9, 0.99], [0.9, 1.0, 0.9], [0.99, 0.9, 1.0]])
    K2 = np.array([[1.0, 0.9, 0.99], [0.9, 1.0, 0.9], [0.99, 0.9, 1.0]])
    K3 = np.array([[1.0, -0.9, 0.9], [-0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    Z_long = (np.transpose(Z, (0, 2, 3, 1))).reshape((N, -1))
    K = Z_long @ np.kron(K3, np.kron(K2, K1)) @ Z_long.T
    y = np.random.multivariate_normal(mean=np.zeros(N), cov=K) + np.random.normal(loc=0, scale=param['sigma'], size=N)

    # create a data object
    data = types.SimpleNamespace()
    data.X, data.Z, data.y, data.A, data.B, data.sig_channel = X, Z, y, A, B, X_channel
    data.K1, data.K2, data.K3 = K1, K2, K3

    # do train-test split
    train_id = math.ceil(train_ratio * N)
    data.train_X, data.test_X, = data.X[0:train_id, :, :, :], data.X[train_id:N, :, :, :]
    data.train_y, data.test_y = data.y[0:train_id], data.y[train_id:N]

    return data