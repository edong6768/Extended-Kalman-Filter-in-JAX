# Extended Kalman Filter in JAX
JAX implementation of Extended Kalman Filter(EKF). JAX is used for its ability to automatically calculate jacobian through `jax.numpy.jacfwd`. There is also an EKF code written with Numpy instead of JAX, where jacobian matrix has to be calculated by hand and then be explicitly given as argument in the code. This code was created as an assignment from lecture DEEE0728-001(Statistical inference and machine learning), KNU Fall 2022.

![](README_asset/track.PNG)

## Dependencies
- jax 0.3.14
- matplotlib 3.5.3
- tqdm
- numpy (for numpy version of EKF)

## Mechanism of Extended Kalman Filter
The extended Kalman filter (EKF) is a widely used algorithm for estimating the state of a dynamic system when the system and measurement models are non-linear. It allows for the incorporation of non-linear models into the Kalman filter framework by linearizing the models around the current estimate of the state.

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

### extended Kalman Filter algorithm
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
