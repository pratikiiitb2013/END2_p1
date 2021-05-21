
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
* Implemented __'__getitem__ function such that it returns image matrix(28X28), on the fly generated random no between 0-9, image label and sum of label and randon no.
* The sum will be between 0-18. This information will be handy later in designing outout layer of NN.
* 


```
options: {
  helpers: 'src/helpers/helper-*.js'
}
```
