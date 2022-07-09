# Stereo-Image
Implementation of Stereo Image for depth estimation using Viterbi and Semi-Global matching algorithms

# Program Description
## SSDD Algorithm
1. Pad left image with zeros (row_pad = window size /2, col_pad = window size /2 + disparity range) same padding size at each side.
2. Loop over disparity labels
   - Pad right image with zeros:<br /> 
    (row_pad = window size /2, col_pad = (window size/2 - current disparity + disparity range),(window size/2 + current disparity + disparity range)<br />
    every iteration the padding is changing for col axis, different padding from left and right sides.
   - Compute (left_pad_image – right_pad_image) ^ 2.
   - Compute 2D convolution on the SSD result with kernel of ones and sum the result of all channels (R, G, B).
  ![image](https://user-images.githubusercontent.com/108329249/178107254-4d72e948-a14d-47b8-83e1-29f468d02c8e.png)
   - Insert result to the SSDD tensor according to the current label
3. Return normalized SSDD tensor

## Naive Depth Map
![image](https://user-images.githubusercontent.com/108329249/178107623-3b6f9eca-a007-4519-90c4-e5308e850db9.png)
<br />As you can see the output is too noisy and not smooth, due to presence of outliers the disparity function can look like:<br />
![image](https://user-images.githubusercontent.com/108329249/178107662-9e69aada-ca25-4f51-b321-e65e71c734a0.png)
<br />There are some local minimums and the algorithm do not know to choose which disparity value is preferable, so the output is not smooth.

## Viterbi Algorithm
The score method for a single slice of the ssdd tensor, using Dynamic Programming.<br />
Where the score for each value is given by:<br />
<br />𝐿(𝑑, 𝑐𝑜𝑙) = 𝐶𝑠𝑙𝑖𝑐𝑒(𝑑, 𝑐𝑜𝑙) + 𝑀(𝑑, 𝑐𝑜𝑙) − 𝑚𝑖𝑛{𝐿(:, 𝑐𝑜𝑙 − 1)}<br />
where:<br />
1. d, cols are the row and column indices in the slice.
2. Note that we “normalize” each column with the minimal value in the previous
column.
3. M(d, col) is the cost of the optimal route until the current location. It is chosen to be
the minimal value of the following items:
    - The score of route L from the previous column for the same d value: L(d, col-1)
    - The score of the optimal L from the previous column with disparity value deviating by ± 1, in addition to a penalty p1:<br />
      𝑃1 + 𝑚𝑖𝑛{𝐿(𝑑 − 1, 𝑐𝑜𝑙 − 1), 𝐿(𝑑 + 1, 𝑐𝑜𝑙 − 1)}
    -  The score of the optimal L from the previous column with disparity value deviating by more than 2, in addition to a penalty p2:<br />
       𝑃2 + 𝑚𝑖𝑛{𝐿(𝑑 + 𝑘1, 𝑐𝑜𝑙 − 1) ∀𝑘: |𝑘| ≥ 2}
       
## Smooth Depth Map - DP
![image](https://user-images.githubusercontent.com/108329249/178107963-4c793ba2-efef-4e6c-811a-6b796608292e.png)
<br />As you can see the output is smoother compared to Naive Depth Map.<br />
This result is unsurprising due to the semi-global approach (Viterbi algorithm) that take care of the value of close pixels.<br />
Viterbi assumption based on the idea that each pixel disparity should be close to its neighbor in order to receive smooth result.

## Semi-Global Matching Algorithm
1.	For each direction we calculated a scored ssdd tensor based on DP (Viterbi algorithm) by the function <br />
   calculate_score_per_direction(ssdd_tensor, direction, p1, p2)
      - According to the given direction we are getting the route which contains all the slice indexes of the tensor for a given direction,<br />
        by the function get_score_route(h, w, direction) 
         - For each slice index in the route we are extracting the corresponding slice per direction by the function <br /> 
            extract_slice_by_direction(ssdd_tensor, slice_index, direction)
         -	Calculate the scored slice based on the Dynamic Programing method.
         -	Insert the scored slice to the directed score tensor.
       - Return the scored tensor corresponding to the given direction.
2.	Compute the average of all scored tensor cross direction.
3.	Return the argmin label of the scored tensor.

### Depth Map Per Direction
![image](https://user-images.githubusercontent.com/108329249/178108477-b9fce515-588c-4cc4-bd69-b51053c23ba7.png)
<br />As you can see, the image for each direction is similar but not exactly the same.<br />
Moreover, you can see that the color "stretches" by the calculation direction.

### Smooth Depth - SGM (all direction combined)
![image](https://user-images.githubusercontent.com/108329249/178108585-9a2b2106-5e6a-40f9-a1ba-101fa0a93acd.png)
<br />As you can see the output is awesome!<br />
We can reconstruct the lamp, the statue, the camera, and the object on the table.<br />
However, can lose the information of the library and the background, and we can't see the facial features of the statue.









