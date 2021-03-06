## END 2 Phase 1 Assignment 2 - working backpropagation on excel
------------------------------------------------------------------------------------------------------------

## Group : 
1. Sunny Sinha
2. Pratik Jain
3. Anudeep
4. MS

----------------------
## Notes 
---------------------------------------------------------------------------------------------------------------------------

Neural net created in Excel sheet with Learning Rate = 0.5

![alt text](https://github.com/pratikiiitb2013/END2_p1/blob/main/Session2/Screenshots/LR_05_Scr_1.png)
![alt_test](https://github.com/pratikiiitb2013/END2_p1/blob/main/Session2/Screenshots/LR_05_Scr_2.png)

Variation in Error graph when we change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 
![alt text](https://github.com/pratikiiitb2013/END2_p1/blob/main/Session2/Screenshots/LR_01_Graph.png)
![alt text](https://github.com/pratikiiitb2013/END2_p1/blob/main/Session2/Screenshots/LR_02_Graph.png)
![alt text](https://github.com/pratikiiitb2013/END2_p1/blob/main/Session2/Screenshots/LR_05_Graph.png)
![alt text](https://github.com/pratikiiitb2013/END2_p1/blob/main/Session2/Screenshots/LR_08_Graph.png)
![alt text](https://github.com/pratikiiitb2013/END2_p1/blob/main/Session2/Screenshots/LR_1_Graph.png)
![alt text](https://github.com/pratikiiitb2013/END2_p1/blob/main/Session2/Screenshots/LR_2_Graph.png)

Major Steps:
  - Design the neural network as shown in above screenshot, with the given inputs, targets and initial weights.
  - Next major step is to calculate the h1,h2, and the activated h1 and h2 that is nothing but sigmoid of h1 and h2 respectively as we are using activation function as Sigmoid.
  - similarly we calulate o1,o2 and the coressponding activated values.
  - Now the next step is to understand that total Error is the combination of two components coming from two branches and is denoted by E1 and E2.
  - This E1 and E2 is the Mean square error, that is calulated as the square of the difference between expected output and actual output along the two branches. Here 1/2 is included just for simplicity of calulation while taking derivatives.
  - Following equations can be derived :
      h1 = w1*i1+w2*i2
      h2 = w3*i1+w4*i2
      a_h1 = ??(h1) = 1/(1+exp(-h1))
      a_h2 = ??(h2)
      o1 = w5*a_h1+w6*a_h2
      o2 = w7*a_h1+w8*a_h2
      a_o1 = ??(o1)
      a_o2 = ??(o2)
      E1 = ??*(t1 - a_o1)??
      E2 = ??*(t2 - a_o2)??
      E_Total = E1 + E2
      
  - Now with backpropagation we move in backward direction, calculating the rate of change in total error with respect individual weights keeping other weights as constant.
  - We first calculate ??E_t/??w5, here we see that w5 has impact only along the path E1 and by using chain rule we can rewrite the equation as : (??E1/??a_o1)*(??a_o1/??o1)*(??o1/??w5)
  - We then calculate each of the components of this equation separately. 
  - And finally we combine the parts together to get : ??E_t/??w5 = (a_o1-t1)*(a_o1*(1-ao1))*a_h1
  - And carefully observing the paths for each of the following weights we can rewrite equation for them as :
        ??E_t/??w6 = (a_o1-t1)*(a_o1*(1-ao1))*a_h2
        ??E_t/??w7 = (a_o2-t2)*(a_o2*(1-ao2))*a_h1
        ??E_t/??w8 = (a_o2-t2)*(a_o2*(1-ao2))*a_h2
  - Now moving further in backpropagation , we see the requirement to calculate similar rate of change of total error with respect to other weights.
  - For this we have to get ??E_t/??w1 , before calculating it directly we need to get equation for ??E_t/??a_h1
  - This a_h1 will have impact along both the paths, ie along the weight w5 and w7, so the impact of a_h1 is distributed along E1 and E2.
  - For calculating ??E_t/??a_h1 , we have to combine the equation for ??E1/??a_h1 and ??E2/??a_h1
  - From the chain rule we can rewrite the equation as ??E1/??a_h1 = (??E1/??a_o1)*(??a_o1/??o1)*(??o1/??a_h1)
  - Above one on simplification gives (a_o1 - t1)*(a_o1*(1-a_o1))*w5
  - From the chain rule we can rewrite the equation as ??E2/??a_h1 = (??E2/??a_o2)*(??a_o2/??o2)*(??o2/a_h1)
  - Above one on simplification gives (a_o2-t2)*(a_o2*(1-a_o2))*w7
  - So, we get ??E_t/??a_h1  = ((a_o1 - t1)*(a_o1*(1-a_o1))*w5) + ((a_o2-t2)*(a_o2*(1-a_o2))*w7)
  - Now to calculate ??E_t/??w1 we can again use the chain rule and rewrite equation as ??E_t/??w1 = (??E_t/??a_o1)*(??a_o1/??o1)*(??o1/??a_h1)*(??a_h1/??h1)*(??h1/??w1)
  - If we see carefully then the combination of first three components on RHS is nothing but ??E_t/??a_h1, so we can rewrite the equation as :
         ??E_t/??w1 = (??E_t/??a_h1)*(??a_h1/??h1)*(??h1/??w1) =  (??E_t/??a_h1)*(a_h1*(1-a_h1))*(i1)  , or by putting above calculated value for ??E_t/??a_h1 in equation we get
         ??E_t/??w1 =( ((a_o1 - t1)*(a_o1*(1-a_o1))*w5)+((a_o2-t2)*(a_o2*(1-a_o2))*w7)*(a_h1*(1-a_h1))*(i1)
  - Similarly we can derive equations by considering the path they affect :
         ??E_t/??w2 = (??E_t/??a_h1)*(a_h1*(1-a_h1))*(i2)
         ??E_t/??w3 = (??E_t/??a_h2)*(a_h2*(1-a_h2))*(i1)
         ??E_t/??w4 = (??E_t/??a_h2)*(a_h2*(1-a_h2))*(i2)
  - For above equations of ??E_t/??w3 and ??E_t/??w4 we need calculation of ??E_t/??a_h2
  - Again we can simplify ??E_t/??a_h2 as ??(E1+E2)/??a_h2
  - Using the chain rule we can rewrite above two terms as :
         ??E1/??a_h2 = (??E1/??a_o1)*(??a_o1/??o1)*(??o1/??a_h2), in this we have already calculated ??E1/??a_o1 and ??a_o1/??o1 previously for ??E1/??a_h1
         ??E2/??a_h2 = (??E2/??a_o2)*(??a_o2/??o2)*(??o2/??a_h2), in this we have already calculated ??E2/??a_o2 and ??a_o2/??o2 previously for ??E2/??a_h1
  - so we can rewrite above equations as :
         ??E1/??a_h2 =  (a_o1-t1)*(a_o1*(1-a_o1))*w6
         ??E2/??a_h2 =  (a_o2-t2)*(a_o2*(1-a_o2))*w8
  - Combining both we get , 
         ??E_t/??a_h2 = ((a_o1-t1)*(a_o1*(1-a_o1))*w6) + ((a_o2-t2)*(a_o2*(1-a_o2))*w8)
  - Now when creating the table given inputs, targets and initial weights are populated, calculated value of h1, h2, a_h1, a_h2, o1, o2, a_o1, a_o2, E1, E2, E_Total, ??E_t/??w1,	??E_t/??w2,	??E_t/??w3,	??E_t/??w4,	??E_t/??w5,	??E_t/??w6,	??E_t/??w7, and	??E_t/??w8.
  - Then we calculate the weights for next phase, going in backpropagation style and starting from w5, the new w5(new) = w5(old) - ??*(??E_t/??w5), here ?? is learning rate.
  - Then we calculate the new weights for others as well in same way, we calculate w6(new), w7(new), w8(new) and in next step we calculate w1(new), w2(new), w3(new) and w4(new).
  - We then extend this table till a significant number of levels so that we could observe that E_Total is decreasing with each level and activated outputs a_o1 and a_o2 move more closure to the expected target t1 and t2.
  - Select the column for E_Total and create a graph to observe the pattern of decreasing E_Total.
 
 Note: We have used derivative of sigmoid function  as :
       {1/(1+exp(-x))}*{1 - (1/(1+exp(-x)))} = ??(x)*(1-??(x))
  
