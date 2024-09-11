### Equations Used in HOG Feature Extraction

1. **Gradient in X and Y directions**:

   $$ 
   G_x = I[i, j+1] - I[i, j-1] 
   $$


   $$ 
   G_y = I[i+1, j] - I[i-1, j] 
   $$


2. **Gradient Magnitude**:


   $$ 
   M(i, j) = \sqrt{G_x^2 + G_y^2} 
   $$


3. **Gradient Angle**:


   $$ 
   \theta(i, j) = \arctan\left(\frac{G_y}{G_x}\right) 
   $$


4. **Histogram Binning**:


   $$ 
   \text{bin}_k = \sum_{(i, j) \in \text{cell}} M(i, j) \cdot \mathbf{1}_{\{B_k \leq \theta(i, j) < B_{k+1}\}} 
   $$


