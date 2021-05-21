
## END 2 Phase 1 Assignment 3 - Pytorch basics and multi-output DNN
------------------------------------------------------------------------------------------------------------

## Group : 
1. Sunny Sinha
2. Pratik Jain
3. Anudeep
4. MS

----------------------
## Notes 
---------------------------------------------------------------------------------------------------------------------------

## Question
* Design a NN that takes 2 inputs and 2 outputs.
* 2 inputs include 1)MNIST image 2)a random no between 0-9.
* 2 outputs include 1)number that is represented by MNIST image  2)sum of MNIST image number and random no input

## Data representation and generation strategy
* Created a custom dataset class inherited from __torch Dataset__ class.
* Passed the MNIST data downloaded csv file to be used in generating data.
* Implemented __getitem__ function such that it returns image matrix(28X28), on the fly generated random no between 0-9, image label and sum of label and random no.
* The sum will be between 0-18. This information will be handy later in designing outout layer of NN.
* Please refer below the code for the class.


```python
transform=transforms.Compose([
    transforms.ToTensor()
])

class customMNISTDataset(Dataset):
  def __init__(self, data, transforms=None):
    self.X = data.iloc[:, 1:]
    self.y = data.iloc[:, 0]
    self.transforms = transforms

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    img_data = self.X.iloc[idx, :]
    img_data = np.array(img_data).astype(np.uint8).reshape(28, 28, 1)
    rand_no = randint(0, 9)
    y1 = self.y[idx]
    y2 = y1 + rand_no
    
    if self.transforms:
      img_data = self.transforms(img_data)
    return img_data, rand_no, y1, y2
```

## Network architecture details(how we are combining the 2 inputs)
* Taking 2 seperate inputs, 1)28X28 tensor for image 2)one hot encoded vector for random no.
* One hot vector will be of size 10 as the no can be between 0-9.
* Processed the image part through combination of conv layers and max pool layers. Then flatten into 1D vector. Finally processed through fully connected layers to create embedding of size 1X5.
* Processe the one hot encoded input of randon no through 2 fully connected layers to create embedding of size 1X5.
* Now there are 2 embeddings representing image and random no. Each of size 1X5.
* Now, these are concatenated to create embedding of size 1X10. Now this concatenated vector represents combination of 2 inputs.
* We further processed this through few fully connected layers to create an output of size 29.
* __Why 29?__ Because we need 2 outputs one if image no and anther is for sum. Imgae no can vary between 0-9 and sum can vary between 0-18. So 10+19 = 29 output size.
* Now this 29 output is divided into 10 and 19 and returned from network. These are used to check against label and sum and train the network.
* For training, the 2 outputs are seperately checked against target values and 2 losses will be created.
* After that sum of losses will be used to calculate the gradients and train the network.

## Evaluation
* After training we have evaluated agaist MNIST test data.
* Following are the results

## Loss function
* Since we are checking 2 outputs and training the model, we will be using 
