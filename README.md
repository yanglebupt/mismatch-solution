# Solve Mismatch Problem in Compressed Sensing

This repo contains the official implementation for the paper ***Solve Mismatch Problem in Compressed Sensing***.

by Le Yang (2019212184@bupt.edu.cn).

## Abstract

This article proposes a novel algorithm for solving mismatch problem in compressed sensing. Its core is to transform mismatch problem into matched by
constructing a new measurement matrix to match measurement value under
unknown measurement matrix. Therefore, we propose mismatch equation and
establish two types of algorithm based on it, which are matched solution of
unknown measurement matrix and calibration of unknown measurement matrix. Experiments have shown that when under low gaussian noise levels, the
constructed measurement matrix can transform the mismatch problem into
matched and recover original images.

## Method

We provide three algorithms for constructing measurement matrix $A_{recv}$ to convert mismatched problem into matched problem.

|  Algo.1   | Algo.2  | Calibration.M |
|  :----:  | :----:  | :----: |
| ![Algo.1](./assets/algo1.png)  | ![Algo.2](./assets/algo2.png) | ![Calibration.M](./assets/cm.png) |

Compressed sensing algorithm used in experiments is $GPSR$

## Results
Mismatched Restored images
![mismatch_recv_res](./results/mismatch_recv_res.jpg)

Iterative loss curves using different `3` $PM_{image}$ for `Algo.1(left)` and `Algo.2(right)`
|||
|  :----:  | :----:  |
| ![Algo.1](./results/exp0/algo1-curve.jpg)  | ![Algo.2](./results/exp0/algo2-curve.jpg) |

Restored `Baboon` image using constructed $A_{recv}$ by `Algo.1` and `Algo.2` with different `3` $PM_{image}$ seee
![PMs](./assets/PMs.png)

Restored images using constructed $A_{recv}$ by `Algo.1(top)` and `Algo.2(Bottom)` with $PM3$ seee
![recv_res_row1_1e-4](./results/exp1/recv_res_row1_1e-4.jpg)

Restored images of different noise levels using `Algo.2` with $PM3$ seee
![recv_res_algo2](./results/exp2-x/recv_res_algo2.jpg)

Restored images of different noise levels using **Calibration of Unknown Measurement Matrix Algorithm**
![recv-calibrationM](./results/calibrationM/recv_res-r.jpg)


## Running Experiments

### Environment

We use the code experiment environment conditions as shown in the following list:

- PyTorch  1.11.0
- Python  3.8 (ubuntu20.04)
- Cuda  11.3
- RTX 2080 Ti (11GB) * 1

### Project structure

```bash
/
├── mmf_displacement # empty folder needs to be replaced with different displacement MMF measurement matrixs downloaded from google drive in the subsection <Data availability>
│   ├── 0 # measurement matrix in 0 displacement MMF
│   │   ├── A_500_256_1.mat  # mmf speckle measurement matrix
│   │   ├── GI_x0y0.mat      # GI Original Image
│   │   ├── y_500_256_1.mat  # Experimental bucket detector value
│   │   └── y_original_500_256_1.mat # Original experimental bucket detector value (before sum)
│   ├── 10 # measurement matrix in 10 displacement MMF. We didn't use in our experiments
│   └── 25 # measurement matrix in 25 displacement MMF
├── timg  # folder including test images and pre-measure images
├── trad_cs_recv_algos # folder containing pytorch implementation of two traditional compressed sensing algorithms, which are OMP and GPSR
│   │── __init__.py  # export module and include a method to get dct transform matrix for sparse transformation
│   │── GPSR_Basic.py # GPSR algorithm for compressed sensing. We used in our experiments
│   └── OPM_recv.py  # OMP algorithm for compressed sensing. We didn't use in our experiments
├── results # empty folder needs to be replaced with results of exps downloaded from google drive in the subsection <Results of Exps>. Or you can run code to save results of exps
├── __init__.py # core implementation of Mismatch Equation and Iterative Algorithm in the paper
├── mmf_speckle.py  # read test images, pre-measure images, and different displacement MMF measurement matrixs
├── recv-mismatch.ipynb # reconstruction of mismatched pairs input into GPSR
├── without-nosie.ipynb # Exp0, Exp1 of algo.1 and algo.2
├── nosie.ipynb # Exp2 of algo.1 and algo.2
├── recv-exps.ipynb # reconstruction of matched pairs constructed by Exp0,Exp1,Exp2 input into GPSR
├── calibration-M.ipynb # implementation and reconstruction of Exp3, which algorithm in the <<B. Calibration of unknown measurement matrix——B.2 Unknow Images in M-Space>> of paper
├── calibration-N.ipynb # implementation and reconstruction of algorithm in the <<B. Calibration of unknown measurement matrix——B.1 Unknow Images in N-Space>> of paper
├── multiply-test.ipynb # show Multiplicity Property of constructed measurement matrix A_recv 
├── README.md
```

## Data availability

Different displacement MMF measurement matrixs at https://drive.google.com/drive/folders/1_RlwkPU6pSR6FRqWL7TT7ovwtphenpcy?usp=drive_link. Download and replace the `/mmf_displacement` folder.

## Results of Exps

We did four different experiments in the paper and saved their constructed measurement matrix $A_{recv}$ and corresponding measurement value $y_u$ of the unkown measurement matrix $A_u$ at 【https://drive.google.com/drive/folders/1_RlwkPU6pSR6FRqWL7TT7ovwtphenpcy?usp=drive_link】. You can download it and replace the `/results` folder. Then you can run $GPSR$ algorithm in `recv-exps.ipynb` and `calibration-M.ipynb?#recv`. You can also try other compressed sensing algorithms to reconstruct original images.

## Notes

We suggest that you have **sufficient storage** to store the various results of constructed $A_{recv}$ when running the experiments. **Otherwise, you need to change the code to an unsaved form and directly use the constructed** $A_{recv}$ **to restore images and show visualization results**.

## References

If you find the code/idea useful for your research, please consider citing

```bib
```
