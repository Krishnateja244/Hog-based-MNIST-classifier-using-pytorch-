import numpy as np 
import matplotlib.pyplot as plt
import torch
import shutil
import seaborn as sns


def plot_confusion_matrix(con_mat):
    """
    Function that plots the confusion matrix

    Args:
        con_mat (np.array): confusion matrix from data
    """
    group_counts = ["{0:0.0f}".format(value) for value in con_mat.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in con_mat.flatten()/np.sum(con_mat)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]
    labels = np.array(labels).reshape(10,10)
    plt.figure(figsize=(12,10))
    ax = sns.heatmap(con_mat, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Mnist Classification Confusion matrix\n\n')
    ax.set_xlabel('\nPredicted labels')
    ax.set_ylabel('Actual lables')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(np.arange(0,10,1))
    ax.yaxis.set_ticklabels(np.arange(0,10,1))

    ## Display the visualization of the Confusion Matrix.
    plt.savefig("./results/test/confusion_matrix.png") 


def plot_loss(steps,train_loss,val_loss,counter):
    """
    Function to plot the training loss and validation loss

    Args:
        steps (int): Number of epoches the training performed
        train_loss (list): training loss at each epoch
        val_loss (list): validation loss at each epoch
        counter (int): integer counter to keep track of set of hyperparameters used
    """
    steps = np.arange(0,len(train_loss),1)
    fig = plt.figure()
    plt.plot(steps,train_loss,label="Traing_loss")
    plt.plot(steps,val_loss,label="Validation_loss")
    plt.title("Tran_loss vs val_loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig(f"./results/validation/train_loss_{str(counter)}.png")

def save_checkpoint(model,checkpoint_path):
    """
    Function to save checkpoints of model

    Args:
        model : model used for training
        is_best (bool): boolean to save at differnt location
        checkpoint_path (str): destination to save checkpoint
        cntr (int): integer counter to keep track of set of hyperparameters used
    """
    best_name = "best_model.pt"
    # if is_best:
    torch.save(model,checkpoint_path+best_name)
        

def display_predictions(model,x_data,x_test):
    """
    Function to plot few predictions on the test dataset

    Args:
        model : best model from training
        x_data (np.array): input training data
        x_test (np.array): input test data
    """
    num_images = 8
    plt.figure()
    rand_indxs = np.random.randint(0,len(x_data),num_images)
    for i in range(len(rand_indxs)):
        with torch.no_grad():
            output = model(x_data[rand_indxs[i]].reshape(1,-1).float())
            _,predicted = torch.max(output.data,1)
        plt.subplot(2,4,i+1)
        plt.imshow(x_test[rand_indxs[i]])
        plt.title(f"Prediction :{predicted.item()}" )
    plt.savefig("./results/test/prediction.png")

def display_train_samples(x_data):
    """
    Function to display the training images 

    Args:
        x_data (np.array): input images of the training data
    """
    num_images = 8
    plt.figure()
    rand_indxs = np.random.randint(0,len(x_data),num_images)
    for i in range(len(rand_indxs)):
        plt.subplot(2,4,i+1)
        plt.imshow(x_data[rand_indxs[i]])
    plt.savefig("./results/train/samples.png")

def load_checkpoint(checkpoint_path):
    """
    Function to load the checkpoint

    Args:
        checkpoint_path (str): destination of the checkpoint

    Returns:
        data from the loaded checkpoint
    """
    check_point = torch.load(checkpoint_path)
    cell_size = check_point["cell_size"]
    block_size = check_point["block_size"]
    bin_size = check_point["bin_size"]
    x_test = check_point["x_test"]
    y_test = check_point["y_test"]
    cntr = check_point["counter"]
    return cell_size,block_size,bin_size,x_test,y_test,check_point,cntr

