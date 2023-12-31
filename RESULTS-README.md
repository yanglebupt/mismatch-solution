## Introduction

This file is used to introduce the directory details for saving experimental construction results.

### File Type

You can use `h5py` module to read the file as dataset from `.h5` files containing two main datasets which are constructed measurement matrix $A_{recv}$ and corresponding measurement value $y_u$ of the unkown measurement matrix $A_u$. You can see the key of datasets in `.h5` file as follows

```python
f = h5py.File("xx.h5","r")
print(f.keys())
```

## Results structure

### Exps

Both include **Algorithm.1** `(algo1)` and **Algorithm.2** `(algo2)`, so only `exp{}/algo1` will be explained below, `exp{}/algo2` has the same structure as it

- `exp0/algo1` include construction results for `Baboon` image using three different $PM_{image}$. The key of datasets are `A_recv` and `y_u`.
  - `PM1.h5`
  - `PM2.h5`
  - `PM3.h5`

- `exp1/algo1` include construction results for `7` images using $PM3$. The key of datasets are `A_recv` and `y_u` 

  - `GI.h5`
  - `Baboon.h5`
  - `Peppers.h5`
  - `Goldhill.h5`
  - `Barbara.h5`
  - `Cameraman.h5`
  - `Lena.h5`

- `exp2/algo1` include construction results with different noise levels $\sigma(0,1,5,10)$ for `GI` and `Baboon` images and use $PM3$. The key of datasets are `A_recv` and `y_u` 
  - `0`
    - `GI.h5`
    - `Baboon.h5`
  - `1`
  - `5`
  - `10`

- `exp2-x/algo1` include construction results with different noise levels $\sigma(0,0.5,1,1.5,2,5)$ for `7` images and use $PM3$, which is an extension of `exp2/algo1`. The key of datasets are `A_recv` and `y_u`.
  - `0`
    - `GI.h5`
    - `Baboon.h5`
    - `Peppers.h5`
    - `Goldhill.h5`
    - `Barbara.h5`
    - `Cameraman.h5`
    - `Lena.h5`
  - `0.5`
  - `1`
  - `1.5`
  - `2`
  - `5`

### Calibrations
- `calibrationM/A_recvs.h5` include calibration results ***Algorithm 3: Calibration of Unknown Measurement Matrix*** with different noise levels $\sigma=(0,0.5,1,1.5,2)$. The key of datasets are 
  - `nosie_{}.format($\sigma$)` is $A_{recv}$
  - `y_nosie_{}".format($\sigma$)` is $y_u$

- `calibrationN/A_recvs.h5` Abandoned and unused.

### Multiplier Property Test

`mt` folder include results of section **III Experiments B. The Impact of Device Precision**, which constructed $A_{recv}$ and $y_u$ for `algo.1`, `algo.2`, `algo.3` in three different devices
  - `mt_recv_RTX2080.h5`  RTX 2080 Ti (11G) device. Key of datasets are `A_recv_algo1`, `A_recv_algo2`, `A_recv_cm` and `y_u`
  - `mt_recv_RTXA4000.h5` RTX A4000(16G) device
  - `mt_recv_RTX3090.h5`  RTX 3090(24G) device 
  - `mt.h5` Key of datasets are `km_RTX2080`, `km_RTXA4000`, `km_RTX3090`. And shape is `(2500,3)`, which column is the **Multiplier Value** $\mathbf{\vec{\lambda}}$ of `algo.1`, `algo.2`, `algo.3`


