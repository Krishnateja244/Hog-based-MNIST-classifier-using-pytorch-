from train import Linear, compute_hog
import torch 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score,precision_score,recall_score,classification_report


def plot_confusion_matrix(con_mat):
    group_counts = ["{0:0.0f}".format(value) for value in con_mat.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in con_mat.flatten()/np.sum(con_mat)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]
    labels = np.array(labels).reshape(10,10)
    ax = sns.heatmap(con_mat, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Mnist Classification Confusion matrix\n\n')
    ax.set_xlabel('\nPredicted labels')
    ax.set_ylabel('Actual lables')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(np.arange(0,10,1))
    ax.yaxis.set_ticklabels(np.arange(0,10,1))

    ## Display the visualization of the Confusion Matrix.
    plt.show()  

def testing(model,x_data,y_data,loss_function):
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

def display_predictions(model,x_data,x_test):
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
    plt.show()

def load_checkpoint(checkpoint_path):
    check_point = torch.load(checkpoint_path)
    cell_size = check_point["cell_size"]
    block_size = check_point["block_size"]
    bin_size = check_point["bin_size"]
    x_test = check_point["x_test"]
    y_test = check_point["y_test"]
    return cell_size,block_size,bin_size,x_test,y_test,check_point

if __name__ == "__main__":
    checkpoint_path = "./models/best_model.pt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # x_train, y_train, x_test, y_test, x_eval, y_eval, class_names = get_dataset( 'mnist')
    
    cell_size,block_size,bin_size,x_test,y_test,check_point = load_checkpoint(checkpoint_path)
    
    x_te_features = compute_hog(cell_size,block_size,bin_size,x_test)
    x_te = torch.from_numpy(x_te_features).to(device)
    y_te = torch.from_numpy(y_test.reshape(-1,1))
    y_te = y_te.type(torch.LongTensor).to(device)
    model = Linear(x_te.shape[1],10).to(device)
    model.load_state_dict(check_point['sate_dict'])
    # model.load_state_dict(torch.load("model_params.pt"))
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    print("#"*10+"Testing"+"#"*10)
    test_accuracy,test_loss = testing(model,x_te,y_te,loss_function=loss_function)
    print(f"testing accuracy :{test_accuracy}")
    print(f"test_loss: {test_loss}")
    display_predictions(model,x_te,x_test)