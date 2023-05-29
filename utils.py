import random, math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from sunpy.visualization import colormaps as cmp
from sklearn.metrics import confusion_matrix as cmt


# plot the channel-average map with the same colormap range
def plot_channel(data, savefig=False, path=None, cmin=None, cmax=None):
    # panel plot of AIA/HMI channels (slices based on PIL)
    channels = ['94', '131', '171', '193', '211', '304', '335', '1600', 'hmi', 'pil']
    channel_cm = [cmp.cm.sdoaia94, cmp.cm.sdoaia131, cmp.cm.sdoaia171, cmp.cm.sdoaia193,
                  cmp.cm.sdoaia211, cmp.cm.sdoaia304, cmp.cm.sdoaia335, cmp.cm.sdoaia1600,
                  cm.gist_gray, cm.gist_gray]

    # set the canvas for plotting
    fig, ax = plt.subplots(2, 5, figsize=(20, 10))

    for ch in channels:
        plotid = channels.index(ch)
        row, col, datach = plotid // 5, plotid % 5, 'ceaimage_' + ch
        plotdata = data[plotid]

        # set ticks
        yticks = np.arange(0, plotdata.shape[0], 100)
        xticks = np.arange(0, plotdata.shape[1], 100)
        
        if (cmin is None) or (cmax is None):
            sns.heatmap(plotdata, ax=ax[row, col],
                    cmap=channel_cm[plotid], cbar_kws=dict(use_gridspec=False, location="bottom"),
                    xticklabels=xticks, yticklabels=yticks)
        else:
            sns.heatmap(plotdata, ax=ax[row, col],
                    cmap=channel_cm[plotid], cbar_kws=dict(use_gridspec=False, location="bottom"),
                    xticklabels=xticks, yticklabels=yticks, vmin=cmin[ch], vmax=cmax[ch])

        # set the ticks and titles
        ax[row, col].set_xticks(xticks)
        ax[row, col].set_yticks(yticks)

        if ch not in ['hmi', 'pil']:
            ax[row, col].set_title('AIA-' + ch.upper(), size=16)
        else:
            ax[row, col].set_title(ch.upper(), size=16)

    if savefig:
        plt.savefig(path, dpi=600, bbox_inches='tight')        
    return ax


def pad(s):
    return s if len(s) >= 2 else ("0" * (2 - len(s)) + s)



def intsy(m):
    if m[0] == "B":
        return float(m[1:]) * 1e2
    elif m[0] == "C":
        return float(m[1:]) * 1e3
    elif m[0] == "M":
        return float(m[1:]) * 1e4
    elif m[0] == "X":
        return float(m[1:]) * 1e5
    elif m[0] == "A":
        return float(m[1:]) * 1e1
    else:
        return 0.0


def soft_threshold(x, l):
    for i in range(x.shape[0]):
        if x[i] > l:
            x[i] -= l
        elif x[i] < -l:
            x[i] += l
        else:
            x[i] = 0
    return x


def gradient_x(mat):
    # compute \nabla_x of a matrix parameter
    height, width = mat.shape[0], mat.shape[1]
    gx = np.zeros_like(mat)

    for i in range(height):
        for j in range(width):
            if j == (width - 1):
                continue
            else:
                gx[i, j] = mat[i, j + 1] - mat[i, j]
    return gx


def l1_norm(mat):
    # compute l1-norm of a matrix
    ncol = mat.shape[1]
    return np.sum(np.abs(mat)) - np.sum(np.abs(mat[:,ncol-1]))


def block_geom(H, W):
    center_x, center_y = H // 2, W // 2
    h, w = H // 4, W // 4

    # choose the geometric center of the signal
    x = np.random.choice(range(center_x - 1, center_x + 2))
    y = np.random.choice(range(center_y - 1, center_y + 2))
    print((x, y))

    img = np.zeros((H, W))
    img[(x - h):(x + h + 1), (y - w):(y + w + 1)] = np.random.normal(loc=2.0, scale=0.1, size=(2 * h + 1, 2 * w + 1))

    return img + np.random.normal(loc=0.0, scale=0.1, size=(H, W))




# coverage probability comparison
def coverage_probability(model, data):
    
    # train set coverage probability
    pred_sigma, prob = model.y_train_pred_s2, 0

    for i in range(pred_sigma.shape[0]):
        sd = pred_sigma[i,i] + model.sigma**2
        if sd < 0:
            sd = model.sigma**2
        lower_b = model.y_train_pred[i] - 1.96 * np.sqrt(sd)
        upper_b = model.y_train_pred[i] + 1.96 * np.sqrt(sd)
        if lower_b <= data.train_y[i] <= upper_b:
            prob += 1
            
    train_prob = prob / pred_sigma.shape[0]
    
    
    # test set coverage probability
    pred_sigma, prob = model.y_pred_s2, 0

    for i in range(pred_sigma.shape[0]):
        sd = pred_sigma[i,i] + model.sigma**2
        if sd < 0:
            sd = model.sigma**2
            
        lower_b = model.y_pred[i] - 1.96 * np.sqrt(sd)
        upper_b = model.y_pred[i] + 1.96 * np.sqrt(sd)
        if lower_b <= data.test_y[i] <= upper_b:
            prob += 1
            
    prob = prob / pred_sigma.shape[0]    
    return train_prob, prob



# chronological split
class FlareData:
    
    def __init__(self, X, y, train_size = 0.75, chron_split=False):
        N = X.shape[0]
        y = np.log10(y) - 3.5
        if not chron_split:
            train_id = random.sample(list(range(N)), math.ceil(train_size*N))
            test_id = np.array([i for i in range(N) if i not in train_id])
        else:
            ids = list(range(N))
            train_id, test_id = ids[0:math.ceil(train_size*N)], ids[math.ceil(train_size*N):]
        self.X, self.y = X, y
        self.train_X, self.train_y = X[train_id], y[train_id]
        self.test_X, self.test_y = X[test_id], y[test_id]
        self.train_id = train_id
    
    def scale_normalize(self):        
        for c in list(range(10)):
            if not c == 8:
                scale_factor = np.amax(self.train_X[:,c,:,:]) + 1e-5                
            else:
                scale_factor = 5000
            
            self.train_X[:,c,:,:] = self.train_X[:,c,:,:] / scale_factor
            self.test_X[:,c,:,:] = self.test_X[:,c,:,:] / scale_factor
            
            
            
# summary statistics of the model
def model_summary(model, data, model_type="GP"):
    y_train, y_test = data.train_y, data.test_y
    
    if model_type == "GP":        
        y_train_pred, y_test_pred = model.y_train_pred[:,0], model.y_pred[:,0]
        y_train_s2, y_test_s2 = np.diagonal(model.y_train_pred_s2) + model.sigma**2, np.diagonal(model.y_pred_s2) + model.sigma**2
    else:
        y_train_pred, y_test_pred = model.predict(data.train_X), model.predict(data.test_X)
    
    
    train_MSE, test_MSE = np.mean((y_train - y_train_pred) ** 2), np.mean((y_test - y_test_pred)**2) # MSE
    train_R2, test_R2 = (np.corrcoef(y_train, y_train_pred)**2)[0,1], (np.corrcoef(y_test, y_test_pred)**2)[0,1] # R-sqaured
    
    
    # True Skill Statistics
    tn, fp, fn, tp = cmt(y_true=(y_train >= 0.0), y_pred=(y_train_pred >= 0.0)).ravel()
    train_TSS = tp/(tp + fn) - fp/(fp+tn)
    
    tn, fp, fn, tp = cmt(y_true=(y_test >= 0.0), y_pred=(y_test_pred >= 0.0)).ravel()
    test_TSS = tp/(tp + fn) - fp/(fp+tn)
    
    # coverage probability
    if model_type == "GP":
        train_prob, test_prob = 0, 0

        for i in range(y_train_pred.shape[0]):
            lower_bound = y_train_pred[i] - 1.96 * np.sqrt(y_train_s2[i])
            upper_bound = y_train_pred[i] + 1.96 * np.sqrt(y_train_s2[i])

            if lower_bound <= y_train[i] <= upper_bound:
                train_prob += 1

        for i in range(y_test_pred.shape[0]):
            lower_bound = y_test_pred[i] - 1.96 * np.sqrt(y_test_s2[i])
            upper_bound = y_test_pred[i] + 1.96 * np.sqrt(y_test_s2[i])

            if lower_bound <= y_test[i] <= upper_bound:
                test_prob += 1
        
        train_prob, test_prob = 100*train_prob/y_train_pred.shape[0], 100*test_prob/y_test_pred.shape[0]        
    else:
        train_prob, test_prob = None, None
        
    
    metrics = {"Type": ["training", "testing"],
               "MSE": [train_MSE, test_MSE], 
               "R-sqaured": [train_R2, test_R2], 
               "TSS": [train_TSS, test_TSS],
               "Coverage": [train_prob, test_prob]
               }    
    
    return pd.DataFrame(metrics).round(decimals = 4)