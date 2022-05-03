from train import Linear, compute_hog
import torch 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score,precision_score,recall_score,classification_report
from utils import plot_confusion_matrix, display_predictions,load_checkpoint

def testing(model,x_data,y_data,loss_function):
    """
    Function to test the trained model

    Args:
        model : trained model
        x_data (tensor): testing HOG features
        y_data (tensor): testing output data
        loss_function : loss function to calculate loss

    Returns:
        Accuracy: accuracy on the data
        testing_loss: test loss
    """
    total_preds, true_preds = 0,0
    temp_loss = []
    preds = []
    print("entered")
    with torch.no_grad():
        for i in range(len(x_data)):
            output = model(x_data[i].reshape(1,-1).float())
            loss = loss_function(output, y_data[i])
            temp_loss.append(loss.item())
            _,predicted = torch.max(output.data,1)
            preds.append(predicted)
            total_preds = y_data.shape[0]
            true_preds += torch.sum(predicted == y_data[i]).item()
        testing_loss = np.mean(temp_loss)
    accuracy = (true_preds/total_preds)*100
    con_mat = confusion_matrix(y_data,preds)
    plot_confusion_matrix(con_mat)
    # precision = precision_score(y_data,preds,average="weighted")
    # recall = recall_score(y_data,preds,average="weighted")
    # f1 = f1_score(y_data,preds,average="weighted")
    # print(f"Precision : {precision}")
    # print(f"Recall : {recall}")
    # print(f"F1_score : {f1}")
    print(classification_report(y_data,preds))
    return accuracy,testing_loss

if __name__ == "__main__":
    checkpoint_path = "./models/best_model.pt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cell_size,block_size,bin_size,x_test,y_test,check_point,cntr = load_checkpoint(checkpoint_path)
    print(cell_size,block_size,bin_size,cntr)
    x_te_features = compute_hog(cell_size,block_size,bin_size,x_test)
    x_te = torch.from_numpy(x_te_features).to(device)
    y_te = torch.from_numpy(y_test.reshape(-1,1))
    y_te = y_te.type(torch.LongTensor).to(device)
    model = Linear(x_te.shape[1],10).to(device)
    model.load_state_dict(check_point['sate_dict'])
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    print("#"*10+"Testing"+"#"*10)
    test_accuracy,test_loss = testing(model,x_te,y_te,loss_function=loss_function)
    print(f"testing accuracy :{test_accuracy}")
    print(f"test_loss: {test_loss}")
    display_predictions(model,x_te,x_test)