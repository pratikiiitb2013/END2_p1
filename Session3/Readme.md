
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
* After that sum of both losses will be used to calculate the gradients and train the network.
* Following is the network details
```python
def conv_block(input_size, output_size, kernel_size):
    block = nn.Sequential(
        nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size), nn.ReLU(), nn.MaxPool2d((2, 2)),
    )
    return block

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = conv_block(input_size=1, output_size=6, kernel_size=3)
    self.conv2 = conv_block(input_size=6, output_size=12, kernel_size=3)

    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(in_features=12*5*5, out_features=16)
    self.fc2 = nn.Linear(in_features=16, out_features=5)

    self.fc3 = nn.Linear(in_features=10, out_features=10)
    self.fc4 = nn.Linear(in_features=10, out_features=5)

    self.fc5 = nn.Linear(in_features=10, out_features=50)
    self.fc6 = nn.Linear(in_features=50, out_features=29)


  def forward(self, img, ohe):

    img = self.conv1(img)
    img = self.conv2(img)
    img = img.reshape(img.shape[0], -1)  # img = img.reshape(-1, 12*5*5)
    img = self.relu(self.fc1(img))
    img = self.relu(self.fc2(img))

    ohe = self.relu(self.fc3(ohe))
    ohe = self.relu(self.fc4(ohe))
    
    x = torch.cat((img, ohe), dim=1)
    x = self.relu(x)
    x = self.relu(self.fc5(x))
    x = self.fc6(x)
    return x[:,0:10],x[:,10:]
```


## Evaluation
* After training we have evaluated agaist MNIST test data.
* Following are the results
* total_correct(images): 9763/10000 total_correct(sum): 9637/10000

## Loss function
* Since we are checking 2 outputs and training the model, we will be using 2 losses and summing them.
* Two losses are calculated by using __cross entropy__ loss.
* We have picked cross entropy loss because it combines log_softmax and nll loss in single loss. It is better suited when training for classification problem with C classes.
* Now, for first output(image label), cross entropy is best suited because we are training for 0-9 digit images.
* For 2nd part also, we have set up the network in such a way that we are training a classification problem with 19 classes( sum can only be between 0-18).
* Had it be the case that we have trained 2nd part(sum) in pure regresssion setting, then MSE loss would have been better suited.

## Training logs
```python
epoch 0 total_correct(images): 22855 / 60000 total_correct(sum): 5665 / 60000 loss: 7807.050562977791
epoch 1 total_correct(images): 41702 / 60000 total_correct(sum): 5935 / 60000 loss: 5764.25578212738
epoch 2 total_correct(images): 47466 / 60000 total_correct(sum): 6548 / 60000 loss: 4982.4205040335655
epoch 3 total_correct(images): 49880 / 60000 total_correct(sum): 8180 / 60000 loss: 4508.7538805007935
epoch 4 total_correct(images): 51578 / 60000 total_correct(sum): 9660 / 60000 loss: 4060.484685897827
epoch 5 total_correct(images): 52944 / 60000 total_correct(sum): 11323 / 60000 loss: 3693.761387348175
epoch 6 total_correct(images): 54010 / 60000 total_correct(sum): 13394 / 60000 loss: 3389.1274438500404
epoch 7 total_correct(images): 54736 / 60000 total_correct(sum): 16223 / 60000 loss: 3133.316538631916
epoch 8 total_correct(images): 55339 / 60000 total_correct(sum): 18751 / 60000 loss: 2928.797610759735
epoch 9 total_correct(images): 55724 / 60000 total_correct(sum): 21089 / 60000 loss: 2764.721148252487
epoch 10 total_correct(images): 56037 / 60000 total_correct(sum): 23396 / 60000 loss: 2627.238964945078
epoch 11 total_correct(images): 56282 / 60000 total_correct(sum): 25322 / 60000 loss: 2517.0053839981556
epoch 12 total_correct(images): 56448 / 60000 total_correct(sum): 27333 / 60000 loss: 2410.4136097729206
epoch 13 total_correct(images): 56596 / 60000 total_correct(sum): 29004 / 60000 loss: 2321.953174650669
epoch 14 total_correct(images): 56715 / 60000 total_correct(sum): 30734 / 60000 loss: 2240.907633394003
epoch 15 total_correct(images): 56868 / 60000 total_correct(sum): 31989 / 60000 loss: 2158.3192783892155
epoch 16 total_correct(images): 56941 / 60000 total_correct(sum): 33398 / 60000 loss: 2085.970966219902
epoch 17 total_correct(images): 57032 / 60000 total_correct(sum): 34418 / 60000 loss: 2023.8528337478638
epoch 18 total_correct(images): 57097 / 60000 total_correct(sum): 35769 / 60000 loss: 1951.671406775713
epoch 19 total_correct(images): 57146 / 60000 total_correct(sum): 37004 / 60000 loss: 1893.907660678029
epoch 20 total_correct(images): 57249 / 60000 total_correct(sum): 38267 / 60000 loss: 1837.9666854590178
epoch 21 total_correct(images): 57329 / 60000 total_correct(sum): 39222 / 60000 loss: 1780.6781024187803
epoch 22 total_correct(images): 57315 / 60000 total_correct(sum): 40136 / 60000 loss: 1733.0326383560896
epoch 23 total_correct(images): 57437 / 60000 total_correct(sum): 41174 / 60000 loss: 1686.9023640155792
epoch 24 total_correct(images): 57494 / 60000 total_correct(sum): 41940 / 60000 loss: 1634.0426862984896
epoch 25 total_correct(images): 57543 / 60000 total_correct(sum): 42559 / 60000 loss: 1590.0974762141705
epoch 26 total_correct(images): 57543 / 60000 total_correct(sum): 43384 / 60000 loss: 1549.0360897481441
epoch 27 total_correct(images): 57616 / 60000 total_correct(sum): 44046 / 60000 loss: 1511.6813726872206
epoch 28 total_correct(images): 57646 / 60000 total_correct(sum): 44615 / 60000 loss: 1478.2972678989172
epoch 29 total_correct(images): 57712 / 60000 total_correct(sum): 45296 / 60000 loss: 1439.268848143518
epoch 30 total_correct(images): 57745 / 60000 total_correct(sum): 45592 / 60000 loss: 1411.0820704251528
epoch 31 total_correct(images): 57746 / 60000 total_correct(sum): 46022 / 60000 loss: 1378.8201094269753
epoch 32 total_correct(images): 57816 / 60000 total_correct(sum): 46547 / 60000 loss: 1355.6427497267723
epoch 33 total_correct(images): 57849 / 60000 total_correct(sum): 47134 / 60000 loss: 1315.2220765277743
epoch 34 total_correct(images): 57905 / 60000 total_correct(sum): 47686 / 60000 loss: 1283.5130268707871
epoch 35 total_correct(images): 57928 / 60000 total_correct(sum): 48025 / 60000 loss: 1249.1159695610404
epoch 36 total_correct(images): 57923 / 60000 total_correct(sum): 48503 / 60000 loss: 1224.6107589676976
epoch 37 total_correct(images): 57938 / 60000 total_correct(sum): 48948 / 60000 loss: 1197.4583518728614
epoch 38 total_correct(images): 57992 / 60000 total_correct(sum): 49340 / 60000 loss: 1169.957283206284
epoch 39 total_correct(images): 58060 / 60000 total_correct(sum): 49440 / 60000 loss: 1147.0308940410614
epoch 40 total_correct(images): 58060 / 60000 total_correct(sum): 49829 / 60000 loss: 1132.9956077635288
epoch 41 total_correct(images): 58130 / 60000 total_correct(sum): 50154 / 60000 loss: 1101.4419314563274
epoch 42 total_correct(images): 58130 / 60000 total_correct(sum): 50517 / 60000 loss: 1086.4023428298533
epoch 43 total_correct(images): 58171 / 60000 total_correct(sum): 50804 / 60000 loss: 1062.2552783116698
epoch 44 total_correct(images): 58161 / 60000 total_correct(sum): 51038 / 60000 loss: 1040.5664383471012
epoch 45 total_correct(images): 58206 / 60000 total_correct(sum): 51240 / 60000 loss: 1020.2230875939131
epoch 46 total_correct(images): 58164 / 60000 total_correct(sum): 51450 / 60000 loss: 1003.5838406458497
epoch 47 total_correct(images): 58209 / 60000 total_correct(sum): 51741 / 60000 loss: 981.6787337474525
epoch 48 total_correct(images): 58230 / 60000 total_correct(sum): 51976 / 60000 loss: 968.2554481215775
epoch 49 total_correct(images): 58292 / 60000 total_correct(sum): 52096 / 60000 loss: 948.8871622681618
epoch 50 total_correct(images): 58301 / 60000 total_correct(sum): 52357 / 60000 loss: 932.5019534230232
epoch 51 total_correct(images): 58251 / 60000 total_correct(sum): 52437 / 60000 loss: 919.6560643650591
epoch 52 total_correct(images): 58300 / 60000 total_correct(sum): 52724 / 60000 loss: 899.340486805886
epoch 53 total_correct(images): 58351 / 60000 total_correct(sum): 52875 / 60000 loss: 878.7015183418989
epoch 54 total_correct(images): 58352 / 60000 total_correct(sum): 53102 / 60000 loss: 866.5813536606729
epoch 55 total_correct(images): 58343 / 60000 total_correct(sum): 53302 / 60000 loss: 856.159958217293
epoch 56 total_correct(images): 58374 / 60000 total_correct(sum): 53501 / 60000 loss: 830.9526519551873
epoch 57 total_correct(images): 58407 / 60000 total_correct(sum): 53743 / 60000 loss: 816.7696008235216
epoch 58 total_correct(images): 58425 / 60000 total_correct(sum): 53795 / 60000 loss: 804.2502889111638
epoch 59 total_correct(images): 58427 / 60000 total_correct(sum): 54048 / 60000 loss: 792.270838195458
epoch 60 total_correct(images): 58405 / 60000 total_correct(sum): 54155 / 60000 loss: 778.7275005150586
epoch 61 total_correct(images): 58433 / 60000 total_correct(sum): 54283 / 60000 loss: 765.2339031230658
epoch 62 total_correct(images): 58457 / 60000 total_correct(sum): 54549 / 60000 loss: 749.3451024852693
epoch 63 total_correct(images): 58486 / 60000 total_correct(sum): 54702 / 60000 loss: 732.7243029531091
epoch 64 total_correct(images): 58474 / 60000 total_correct(sum): 54823 / 60000 loss: 721.2585965916514
epoch 65 total_correct(images): 58475 / 60000 total_correct(sum): 54994 / 60000 loss: 699.1032366920263
epoch 66 total_correct(images): 58489 / 60000 total_correct(sum): 55055 / 60000 loss: 693.2092374414206
epoch 67 total_correct(images): 58486 / 60000 total_correct(sum): 55364 / 60000 loss: 673.8872648179531
epoch 68 total_correct(images): 58535 / 60000 total_correct(sum): 55592 / 60000 loss: 653.2858147714287
epoch 69 total_correct(images): 58531 / 60000 total_correct(sum): 55800 / 60000 loss: 632.7052873149514
epoch 70 total_correct(images): 58534 / 60000 total_correct(sum): 56059 / 60000 loss: 613.8695504628122
epoch 71 total_correct(images): 58550 / 60000 total_correct(sum): 56178 / 60000 loss: 593.2711629765108
epoch 72 total_correct(images): 58567 / 60000 total_correct(sum): 56236 / 60000 loss: 578.146983970888
epoch 73 total_correct(images): 58601 / 60000 total_correct(sum): 56488 / 60000 loss: 565.0143378758803
epoch 74 total_correct(images): 58604 / 60000 total_correct(sum): 56625 / 60000 loss: 546.0997259346768
epoch 75 total_correct(images): 58614 / 60000 total_correct(sum): 56689 / 60000 loss: 530.0479660760611
epoch 76 total_correct(images): 58601 / 60000 total_correct(sum): 56926 / 60000 loss: 512.3680435298011
epoch 77 total_correct(images): 58623 / 60000 total_correct(sum): 57068 / 60000 loss: 506.598953246139
epoch 78 total_correct(images): 58662 / 60000 total_correct(sum): 57020 / 60000 loss: 494.1625403170474
epoch 79 total_correct(images): 58660 / 60000 total_correct(sum): 57265 / 60000 loss: 477.1597993527539
epoch 80 total_correct(images): 58674 / 60000 total_correct(sum): 57238 / 60000 loss: 467.4792817477137
epoch 81 total_correct(images): 58673 / 60000 total_correct(sum): 57323 / 60000 loss: 462.0934607265517
epoch 82 total_correct(images): 58680 / 60000 total_correct(sum): 57308 / 60000 loss: 452.2327322214842
epoch 83 total_correct(images): 58724 / 60000 total_correct(sum): 57376 / 60000 loss: 442.929118885193
epoch 84 total_correct(images): 58733 / 60000 total_correct(sum): 57465 / 60000 loss: 432.7389331795275
epoch 85 total_correct(images): 58725 / 60000 total_correct(sum): 57505 / 60000 loss: 430.1324847515207
epoch 86 total_correct(images): 58714 / 60000 total_correct(sum): 57626 / 60000 loss: 421.15098451427184
epoch 87 total_correct(images): 58720 / 60000 total_correct(sum): 57602 / 60000 loss: 413.28731849836186
epoch 88 total_correct(images): 58744 / 60000 total_correct(sum): 57632 / 60000 loss: 406.77272561425343
epoch 89 total_correct(images): 58745 / 60000 total_correct(sum): 57650 / 60000 loss: 407.3495928780176
epoch 90 total_correct(images): 58787 / 60000 total_correct(sum): 57699 / 60000 loss: 403.2898214738816
epoch 91 total_correct(images): 58805 / 60000 total_correct(sum): 57705 / 60000 loss: 392.67584449541755
epoch 92 total_correct(images): 58800 / 60000 total_correct(sum): 57769 / 60000 loss: 385.10193525324576
epoch 93 total_correct(images): 58784 / 60000 total_correct(sum): 57814 / 60000 loss: 380.3785708574578
epoch 94 total_correct(images): 58837 / 60000 total_correct(sum): 57850 / 60000 loss: 371.06351172341965
epoch 95 total_correct(images): 58856 / 60000 total_correct(sum): 57866 / 60000 loss: 373.8245074665174
epoch 96 total_correct(images): 58827 / 60000 total_correct(sum): 57892 / 60000 loss: 366.9756800418254
epoch 97 total_correct(images): 58863 / 60000 total_correct(sum): 57904 / 60000 loss: 360.031832030043
epoch 98 total_correct(images): 58855 / 60000 total_correct(sum): 57920 / 60000 loss: 357.20636436296627
epoch 99 total_correct(images): 58859 / 60000 total_correct(sum): 57978 / 60000 loss: 353.28860045969486
```
