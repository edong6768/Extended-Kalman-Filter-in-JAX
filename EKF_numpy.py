#%% model declaration
import numpy as np
from numpy.random import multivariate_normal as mn
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
from typing import Callable
from dataclasses import dataclass

os.environ['KMP_DUPLICATE_LIB_OK']='True'

@dataclass
class NonLinearModel:
    
    smean: np.array
    covQ: np.array
    omean: np.array
    covR: np.array
    f: Callable[[np.array], np.array]
    jf: Callable[[np.array], np.array]
    h: Callable[[np.array], np.array]
    jh: Callable[[np.array], np.array]
    state_0: np.array
        
    def observe(self, states):
        v = mn(mean=np.zeros((2,)), cov=self.covR, size=states.shape[:-1])
        return self.h(states.T).T + v

    def nextstate(self, state_curr):
        w = mn(mean=np.zeros((3,)), cov=self.covQ)
        return self.f(state_curr) + w

    def sample(self, n=100):
        states = [self.state_0]
        for _ in trange(n, desc='Sampling'):
            states.append(self.nextstate(states[-1]))
        return np.stack(states)
    
    
def EKF(model: NonLinearModel, obsvs, xhat0=None, covP0=None):
    """Extended Kalman Filter"""
    xs, Ps = [model.smean if xhat0 is None else xhat0], [model.covQ if covP0 is None else covP0]
    f, jf, Q, h, jh, R = map(model.__getattribute__, ('f', 'jf', 'covQ', 'h', 'jh', 'covR'))
    
    x, P = xs[-1], Ps[-1]
    for y in tqdm(obsvs, desc='Running EKF'):
        # pred
        A = jf(x)
        x_, P_ = f(x), A@P@A.T+Q
        
        # corr
        H = jh(x_)
        K = P_@H.T@np.linalg.inv(H@P_@H.T+R)
        x, P = x_+K@(y-h(x_)), P_-K@H@P_
        
        xs.append(x), Ps.append(P)

    return np.stack(xs), np.stack(Ps)
    

if __name__=='__main__':
    v, dtheta = 100, 3
    model = NonLinearModel(
                smean = np.zeros((3,)),
                covQ = np.diag(np.array((0.001, 0.001, 3))),
                omean = np.zeros((2,)),
                covR = 0.01*np.eye(2),
                f = lambda s: s + np.array([v*np.cos(s[2]),
                                                v*np.sin(s[2]),
                                                dtheta]),
                jf = lambda s: np.eye(3) + np.array([[0, 0, -v*np.sin(s[2])],
                                                     [0, 0, v*np.cos(s[2])],
                                                     [0, 0, 0]]),
                h = lambda s: np.array([np.sqrt(s[0]**2+s[1]**2),
                                        np.arctan2(s[1], s[0])-s[2]]),
                jh = lambda s: np.array([[s[0]/np.sqrt(s[0]**2+s[1]**2), s[1]/np.sqrt(s[0]**2+s[1]**2), 0],
                                         [-s[1]/(s[0]**2+s[1]**2), s[0]/(s[0]**2+s[1]**2), -1]]),
                
                state_0 = np.zeros((3,)),
            )

    states = model.sample(n=200)
    obsvs = model.observe(states)
    preds, Ps = EKF(model, obsvs[1:])

    # plot
    plt.subplot(221),plt.tick_params(labelsize=6)
    plt.plot(*states.T[:2], '.-', label='ground')
    plt.plot(*preds.T[:2], '.-', label='pred')
    plt.plot(*[[0]]*2, 'kx', mew=2)
    plt.legend()

    plt.subplot(222), plt.title('r'), plt.tick_params(labelsize=6)
    *map(plt.plot, (obsvs.T[0], (preds.T[0]**2+ preds.T[1]**2)**0.5)),

    titles = 'x', 'y', 'theta'
    for n, (title, i, ihat) in enumerate(zip(titles, states.T, preds.T)):
        plt.subplot(234+n), plt.title(title), plt.tick_params(labelsize=6)
        *map(plt.plot, (i, ihat)),
    plt.show()


# %%
