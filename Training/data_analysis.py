import random
import matplotlib.pyplot as plt
from numpy import where

def display_errors(img_errors, pred_errors, obs_errors, classes, class_of_interest=None):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 5

    if class_of_interest is not None:
        # Select a specific class
        img_errors = img_errors[where(obs_errors==class_of_interest)]
        pred_errors = pred_errors[where(obs_errors==class_of_interest)]
        obs_errors = obs_errors[where(obs_errors==class_of_interest)]

    errors_idxs = random.sample(range(0, len(img_errors)), len(img_errors))

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True, figsize=(14,10))
    for row in range(nrows):
        for col in range(ncols):
            error = errors_idxs[n]
            ax[row,col].imshow((img_errors[error]).reshape((128,45)), cmap='turbo')
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format( classes[int(pred_errors[error])], classes[int(obs_errors[error])]  ))
            n += 1