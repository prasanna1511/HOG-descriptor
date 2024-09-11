### Equations Used in HOG Feature Extraction

1. **Gradient in X and Y directions**:

   $$ 
   G_x = I[i, j+1] - I[i, j-1] 
   $$

   $$ 
   G_y = I[i+1, j] - I[i-1, j] 
   $$

2. **Gradient Magnitude**:

  
   $M_{ij} = \sqrt{Gx_{ij}^2 + Gy_{ij}^2}$
   

3. **Gradient Angle**:

    $\theta_{ij} = tan^{-1}(\frac{Gy_{ij}}{Gx_{ij}})$

