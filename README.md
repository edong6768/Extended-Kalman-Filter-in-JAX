# Extended Kalman Filter in JAX
JAX implementation of Extended Kalman Filter(EKF). JAX is used for its ability to automatically calculate jacobian through `jax.numpy.jacfwd`. There is also an EKF code written with Numpy instead of JAX, where jacobian matrix has to be calculated by hand and then be explicitly given as argument in the code. This code was created as an assignment from lecture DEEE0728-001(Statistical inference and machine learning), KNU Fall 2022.

![](README_asset/track.PNG)

## Dependencies
- jax 0.3.14
- matplotlib 3.5.3
- tqdm
- numpy (for numpy version of EKF)

## Kalman Filter
The Kalman filter iteratively estimates the state from measurements of linear system and measurement model with additive white gaussian noise(AWGN).

### Prerequisite
#### notations:
<table>
  <tr>
    <th>$x_k$</th>
    <td>state at time k</td>
  </tr>
  <tr>
    <th>$y_k$</th>
    <td>measurement at time k</td>
  </tr>
  <tr>
    <th>$y_{1:k}$</th>
    <td>$y_1, ..., y_k$ (given measurements $y_1\sim y_k$)</td>
  </tr>
  <tr>
    <th>$\hat{x}_{k|k-1}$</th>
    <td>$\hat{x}_k|y_{1:k}$ (estimated $\hat{x}^k$ given $y_1\sim y_k$)</td>
  </tr>
</table>

#### Formulas used:
1. Mean/Variance of random variable $X$ under multiplication of matrix $F$
$$
\begin{equation}
\mathbb{E}[FX] = F(\mathbb{E}[X])
\end{equation}
$$

$$
\begin{equation}
VAR[FX] = F\ VAR[X]\ F^T
\end{equation}
$$

2. Mean/Variance of sum of random variable $X$, $Y$
$$
\begin{equation}
\mathbb{E}[X+Y] = \mathbb{E}[X] + \mathbb{E}[Y]
\end{equation}
$$

$$
\begin{equation}
VAR[X+Y] = VAR[X] + VAR[Y] + 2\ cov(X,Y)
\end{equation}
$$

3. Conditional Gaussian
$$
given \begin{bmatrix}x \\ y \end{bmatrix} \sim \mathcal{N} 
\left( \begin{bmatrix}\mu_x \\ \mu_y \end{bmatrix},
\begin{bmatrix}P_{xx} & P_{xy}\\ P_{yx} & P_{yy} \end{bmatrix}\right)
$$

$$
\begin{equation}
x|y \sim \mathcal{N} \left(\ \mu_x +P_{xy}P_{yy}^{-1}(y-\mu_y),\ 
P_{xx}-P_{xy}P_{yy}^{-1}P_{yx}\ \right)
\end{equation}
$$

### Linear Model
**State Transition**|$x_k = Fx_{k-1} + w_{k}$
|-|-|
**Measurement**|$y_k = Hx_{k} + v_{k}$

where $F$ and $H$ are matrices and distribution of noise terms are given as $w\sim\mathcal{N}(0, Q)$ and $v\sim\mathcal{N}(0, R)$.
Assume no covariance between $w$, $v$ ( $\mathbb{E}[w^Tv]=0$. )

### Kalman Filter Derivation
Kalman filter essentially updates the posterior distribution of the state for each measurements same as **Density Propagation**, then use **Bayesian Optimal Estimator(Posterior Mean)** to estimate the current state of the system.

Let previous posterior as $x_{k-1|k-1} \sim \mathcal{N}(\hat{x}_{k-1|k-1}, P_{k-1|k-1})$
1. Prediction step (before new measurement) :\
    From state transition model
    $$
    x_{k|k-1}=Fx_{k-1|k-1}+w_{k}
    $$
    Using formula $(1)$~$(4)$
    $$
    x_{k|k-1} \sim \mathcal{N}\left( \hat{x}_{k|k-1}, P_{k|k-1}\right)
    = \mathcal{N} \left( F\hat{x}_{k-1|k-1}\ ,\ FP_{k-1|k-1}F^T+Q\right)\\
    $$
2. Correction step (after new measurement) :\
    From measurement model
    $$
    y_{k|k-1}=Hx_{k|k-1}+v_k
    $$
    Using formula $(1)$~$(4)$
    $$
    y_{k|k-1} \sim \mathcal{N}\left( \hat{y}_{k|k-1}, S_k\right)
    = \mathcal{N} \left( H\hat{x}_{k|k-1}\ ,\ HP_{k|k-1}H^T+R \right)
    $$
    Joint distribution of $x_{k|k-1}$ and $y_{k|k-1}$ then will be
    $$
    P\left( \begin{bmatrix}x_k \\ y_k \end{bmatrix} |\ y_{1:k-1} \right) 
    = \mathcal{N}\left( \begin{bmatrix}\hat{x}_{k|k-1} \\ \hat{y}_{k|k-1} \end{bmatrix},\ 
    \begin{bmatrix}P_{k|k-1} & P_{k|k-1}H^T\\\  HP_{k|k-1}^T & S_k\ \end{bmatrix}\right)
    $$
    Using formala $(5)$
    $$
    x_{k|k} \sim \mathcal{N} \left(\ \hat{x}_{k|k-1}+P_{k|k-1}H^TS_k^{-1}(y_k-\hat{y}_{k|k-1}),\ 
    P_{k|k-1}-(P_{k|k-1}H^T)S_k^{-1}(HP_{k|k-1}^T)\ \right)
    $$
    We call term $P_{k|k-1}H^TS_k^{-1}$ a Kalman gain $K_k$. Substituting gives
    $$
    x_{k|k} \sim \mathcal{N} (\hat{x}_{k|k}, P_{k|k})
    $$
    where
    $$
    \hat{x}_{k|k} = \hat{x}_{k|k-1}+K_k(y_k-\hat{y}_{k|k-1})\\
    P_{k|k} = P_{k|k-1}-K_kHP_{k|k-1}
    $$
    Here we discard transpose in $P_{k|k-1}^T$ since covariance matrices are symmetric.

### Kalman Filter Algorithm
At each time step, the KF trys to predict state estimation $\hat{x}_k$ and state covariance matrix $P_k$ through the following steps:

<table>
  <tr>
    <td><strong>Prediction step</strong><br />(before measurement $y_k$)</td>
    <td>Predict $x_k$ given only $y _{1:k-1}$<br />$$\hat{x} _{k|k-1} = F\hat{x} _{k-1|k-1}$$ $$P _{k|k-1} = FP _{k-1|k-1}F^T+Q$$</td>
  </tr>
  <tr>
    <td><strong>Correction step</strong><br />(after measurement $y_k$)</td>
    <td>Correct $x_k$ with additional measurement $y_k$<br />$$\hat{x} _{k|k} = \hat{x} _{k|k-1} + K_k(y_k - H\hat{x} _{k|k-1})$$ $$P _{k|k} = P _{k|k-1}-K_kHP _{k|k-1}$$ where $K_k$ is the Kalman gain at time step $k$, which is given by:<br />$K_k = P_{k|k-1}H^T(HP_{k|k-1}H^T + R)^{-1}$</td>
  </tr>
</table>


## Extended Kalman Filter
The extended Kalman filter (EKF) is a widely used algorithm for estimating the state of a dynamic system when the system and measurement models are non-linear. It allows for the incorporation of non-linear models into the Kalman filter framework by linearizing the models around the current estimate of the state.

### Non-Linear Model

**State Transition**|$x_k = f(x_{k-1}) + w_{k}$
|-|-|
**Measurement**|$y_k = h(x_{k}) + v_{k}$

where $f(\cdot)$ and $h(\cdot)$ are non-linear functions and distribution of noise terms are given as $w\sim\mathcal{N}(0, Q)$ and $v\sim\mathcal{N}(0, R)$. In order to use the Kalman filter, we must linearize this model around the current estimate of the state, $\hat{x}_{k-1}$. 

### Linearize
Consider a non-linear system model of the form:

$$x_k = f(x_{k-1}) + w_{k}$$

This linearized model is given by:

$$ x_k \approx A\hat{x}_{k-1} + w_{k-1} $$

where $A$ is the Jacobian of the non-linear function $f(\cdot)$ evaluated at the current estimate of the state:

$$ A = \frac{\partial f}{\partial x}(\hat{x} _{k-1|k-1}) $$

Similarly, consider a non-linear measurement model of the form:

$$ y_k = h(x_k) + v_k $$

where $h(\cdot)$ is a non-linear function. In order to use the Kalman filter, we must linearize this model around the current estimate of the state, $\hat{x}_k$. This linearized model is given by:

$$ y_k \approx H\hat{x}_k + v_k $$

where $H$ is the Jacobian of the non-linear function $h(\cdot)$ evaluated at the current estimate of the state:

$$ H = \frac{\partial h}{\partial x}(\hat{x} _{k|k-1}) $$

With these linearized models, we can proceed with the Kalman filter as usual, using the prediction and correction steps described below.

### Extended Kalman Filter algorithm
At each time step, the EKF trys to predict state estimation $\hat{x}_k$ and state covariance matrix $P_k$ through the following steps:

<table>
  <tr>
    <td><strong>Prediction step</strong><br />(before measurement $y_k$)</td>
    <td>Predict $x_k$ given only $y _{1:k-1}$<br />$$\hat{x} _{k|k-1} = f(\hat{x} _{k-1|k-1})$$ $$P _{k|k-1} = AP _{k-1|k-1}A^T+Q$$</td>
  </tr>
  <tr>
    <td><strong>Correction step</strong><br />(after measurement $y_k$)</td>
    <td>Correct $x_k$ with additional measurement $y_k$<br />$$\hat{x} _{k|k} = \hat{x} _{k|k-1} + K_k(y_k - h(\hat{x} _{k|k-1}))$$ $$P _{k|k} = P _{k|k-1}-K_kHP _{k|k-1}$$ where $K_k$ is the Kalman gain at time step $k$, which is given by:<br />$K_k = P_{k|k-1}H^T(HP_{k|k-1}H^T + R)^{-1}$</td>
  </tr>
</table>

Here, $A$ is the linearized state transition matrix, $H$ is the linearized measurement matrix, $Q$ is the state transition noise covariance matrix, and $R$ is the measurement noise covariance matrix.

One important consideration when using the extended Kalman filter is the choice of the process and measurement noise covariances, $Q$ and $R$. In the linear Kalman filter, these covariances can be chosen based on the knowledge of the noise characteristics of the system and measurements. However, in the non-linear case, the choice of these covariances can have a significant impact on the performance of the EKF. In general, it is recommended to choose these covariances based on the uncertainty in the linearized models, rather than the uncertainty in the true non-linear models.
