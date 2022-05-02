from scipy import rand
from zmq import device
from dataset import get_dataset
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from utils import plot_loss, save_checkpoint

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
        with torch.no_grad():
            for i in range(len(x_eval)):
                output = model(x_eval[i].reshape(1,-1).float())
                loss = loss_function(output, y_eval[i])
                temp_val_loss.append(loss.item())
            val_loss.append(np.mean(temp_val_loss))
        # if epoch %10 == 0 :
        print(f"Epoch: {epoch}, training_loss: {np.mean(temp_loss)},val_loss: {np.mean(temp_val_loss)}")
    return train_loss,val_loss



# def plot_loss(steps,train_loss,val_loss,counter):
#     steps = np.arange(0,len(train_loss),1)
#     fig = plt.figure()
#     plt.plot(steps,train_loss,label="Traing_loss")
#     plt.plot(steps,val_loss,label="Validation_loss")
#     plt.title("Tran_loss vs val_loss")
#     plt.xlabel("Steps")
#     plt.ylabel("Loss")
#     plt.grid()
#     plt.legend()
#     plt.savefig(f"./results/validation/train_loss_{str(counter)}.png")

# def save_checkpoint(model,is_best,checkpoint_path,cntr):
#     f_name = f"model_param_{str(cntr)}.pt"
#     best_name = "best_model.pt"
#     torch.save(model,checkpoint_path+f_name)
#     if is_best:
#         shutil.copyfile(checkpoint_path+f_name,checkpoint_path+best_name)

def classifier_analyzer(cell_size_params,block_size_params,bins_params,x_train,x_eval,y_train,y_eval):
    all_val_loss = []
    min_val_loss = np.inf
    cntr = 0 
    for i in range(len(cell_size_params)):
        for q in range(len(block_size_params)):
            for r in range(len(bins_params)):
                print(f"cell_size = {cell_size_params[i]}\
                      block_size = {block_size_params[q]}\
                      bin_size={bins_params[r]}")
                
                x_tr_features = compute_hog(cell_size_params[i],block_size_params[q],bins_params[r],x_train)
                x_ev_features = compute_hog(cell_size_params[i],block_size_params[q],bins_params[r],x_eval)

                x_tr = torch.from_numpy(x_tr_features)
                y_tr = torch.from_numpy(y_train.reshape(-1,1))
                y_tr = y_tr.type(torch.LongTensor)

                x_ev = torch.from_numpy(x_ev_features).to(device)
                y_ev= torch.from_numpy(y_eval.reshape(-1,1))
                y_ev = y_ev.type(torch.LongTensor).to(device)

                #initializing the model
                model = Linear(x_tr.shape[1],len(class_names)) 
                loss_function = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters())
                epoches = 2
                #Training
                train_loss, eval_loss = train(model,epoches,x_tr,y_tr,x_ev,y_ev,loss_function,optimizer)
                all_val_loss.append(eval_loss[-1])
                # saving the model
                checkpoint = {
                    "cell_size": cell_size_params[i],
                    "block_size": block_size_params[q],
                    "bin_size": bins_params[r],
                    "x_test": x_test,
                    "y_test": y_test,
                    "min_val_loss": eval_loss,
                    "sate_dict": model.state_dict(),
                    "counter" : cntr 
                    }
                save_checkpoint(checkpoint,False,checkpoint_path,cntr)
                if eval_loss[-1] <= min_val_loss:
                    print(f"validation loss decreased from {min_val_loss} to {eval_loss[-1]}, saving model .....")
                    save_checkpoint(checkpoint,True,checkpoint_path,cntr)
                    plot_loss(epoches,train_loss,eval_loss,cntr)
                    min_val_loss = eval_loss[-1]
                cntr +=1
    
if __name__ == "__main__":
    checkpoint_path = "./models/"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cell_size_params = [(8,8),(4,4)] #,(7,7)]
    block_size_params = [(1,1),(1,2)] #,(2,2)]
    bins_params = [9,8,6]

    x_train, y_train, x_test, y_test, x_eval, y_eval, class_names = get_dataset( 'mnist')
    classifier_analyzer(cell_size_params,block_size_params,bins_params,x_train,x_eval,y_train,y_eval)