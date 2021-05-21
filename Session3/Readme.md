
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


```py
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
