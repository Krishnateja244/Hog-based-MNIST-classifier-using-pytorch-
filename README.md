### Deep learning project seed
Use this seed to start new deep learning / ML projects.

- Built in setup.py
- Built in requirements
- Examples with MNIST
- Badges
- Bibtex

#### Goals  
The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.   

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT  
 
---

<div align="left">    
 
# Hog based-MNIST classifier using pytorch 

-->
<!-- ![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)
 -->

<!--  
Conference   
-->   
</div>
 
## Description  

This Project classifies MNIST dataset consisting of handwritten digits between 0-9 using Histogram of Oriented Gradients(HOG) features. Pytorch is used for building this classifier. MNIST contains 70,000 images of handwritten digits: 60,000 for training and 10,000 for testing. The images are grayscale, 28x28 pixels. 

Nowadays, Convolutional Neural Networks(CNN) are the state of the art for classifying the MNIST dataset. But this project focuses mainly on how HOG parameters influence the feature extraction process and influence the classification. 

## Dataset preparation

The datasets are downloaded from the ``` torchvision.datasets.MNIST ``` and pytorch dataloader ``` torch.utils.data.DataLoader``` to load batches of training data. 

Few images of dataset are visualized here in below figure.

![My Image](results/test/prediction.png)

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash

# module folder
cd project

# run module 
python lit_classifier_main.py    
```

## Imports

```python

```

<!-- ### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```    -->
