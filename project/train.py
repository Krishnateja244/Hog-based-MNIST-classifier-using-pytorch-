from scipy import rand
from zmq import device
from dataset import get_dataset
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sns
import pandas as pd

class Linear(torch.nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.in_features = in_dim
        self.out_dim = out_dim
        self.weights = torch.nn.Parameter(torch.randn(out_dim,in_dim))
        self.bias = torch.nn.Parameter(torch.randn(out_dim))
    
    def forward(self,inputs):
        output = inputs @ self.weights.t() +self.bias
        return output


def compute_hog(cell_size: int, block_size: int, nbins: int, imgs_gray: np.ndarray) -> np.ndarray:
    """
    Wrapper for the OpenCV interface for HOG features.
    
    Parameters
    ----------
    cell_size: int
        number of pixels in a square cell in x and y direction (e.g. (4,4), (8,8))
    block_size int
        number of cells in a block in x and y direction (e.g., (1,1), (1,2))
    nbins: int
        number of bins in a orientation histogram in x and y direction (e.g. 6, 9, 12)
    imgs_gray np.ndarray
        images with which to perform HOG feature extraction (dimensions (nr, width, height))
    
    Returns
    ----------
    np.ndarray
        array of shape H x imgs_gray.shape[0] where H is the size of the resulting HOG feature vector
        (depends on parameters)
    """
    hog = cv2.HOGDescriptor(_winSize=(imgs_gray.shape[2] // cell_size[1] * cell_size[1],
                                      imgs_gray.shape[1] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    # winSize is the size of the image cropped to a multiple of the cell size

    hog_example = hog.compute(np.squeeze(imgs_gray[0, :, :]).astype(np.uint8)).flatten().astype(np.float32)

    hog_feats = np.zeros([imgs_gray.shape[0], hog_example.shape[0]])

    for img_idx in range(imgs_gray.shape[0]):
        hog_image = hog.compute(np.squeeze(imgs_gray[img_idx, :, :]).astype(np.uint8)).flatten().astype(np.float32)
        hog_feats[img_idx, :] = hog_image

    return hog_feats



def train(model,epoches,x_data,y_data,x_eval,y_eval,loss_function,optimizer):
    
    train_loss = []
    val_loss = []
    for epoch in range(epoches):
        temp_loss = []
        for i in range(len(x_data)):
            output = model(x_data[i].reshape(1,-1).float())
            loss = loss_function(output, y_data[i])
            loss.backward()
            temp_loss.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        train_loss.append(np.mean(temp_loss))
        temp_val_loss = []
        total_preds = 0
        with torch.no_grad():
            for i in range(len(x_eval)):
                output = model(x_eval[i].reshape(1,-1).float())
                loss = loss_function(output, y_eval[i])
                temp_val_loss.append(loss.item())
            val_loss.append(np.mean(temp_val_loss))

        # if epoch %10 == 0 :
        print(f"Epoch: {epoch}, training_loss: {np.mean(temp_loss)},val_loss: {np.mean(temp_val_loss)}")

    return train_loss,val_loss



def plot_loss(steps,train_loss,val_loss):
    steps = np.arange(0,len(train_loss),1)
    fig = plt.figure()
    plt.plot(steps,train_loss,label="Traing_loss")
    plt.plot(steps,val_loss,label="Traing_loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    path = "./models/"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x_train, y_train, x_test, y_test, x_eval, y_eval, class_names = get_dataset( 'mnist')
    #computing HOG features for all training images
    x_tr_features = compute_hog([8,8],[1,1],9,x_train)
    #computing HOG features for all test images
    x_ev_features = compute_hog([8,8],[1,1],9,x_eval)

    x_tr = torch.from_numpy(x_tr_features).to(device)
    y_tr = torch.from_numpy(y_train.reshape(-1,1))
    y_tr = y_tr.type(torch.LongTensor).to(device)

    x_ev = torch.from_numpy(x_ev_features).to(device)
    y_ev= torch.from_numpy(y_eval.reshape(-1,1))
    y_ev = y_ev.type(torch.LongTensor).to(device)

    #initializing the model
    model = Linear(x_tr.shape[1],len(class_names)).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    epoches = 21
    #Training
    print("#"*10+" Training "+"#"*10)
    train_loss, eval_loss = train(model,epoches,x_tr,y_tr,x_ev,y_ev,loss_function,optimizer)
    plot_loss(epoches,train_loss,eval_loss)
    # saving the model
    torch.save(model.state_dict(),path+"model_params.pt")