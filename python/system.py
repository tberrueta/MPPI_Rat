from utils import *

def diff_drive(x,u):
    xvel = [u[0]*np.cos(x[2]),
            u[0]*np.sin(x[2]),
            u[1]]
    return np.array(xvel).flatten()

def single_int(x,u):
    xvel = [u[0],
            u[1]]
    return np.array(xvel).flatten()

def quadratic_objective(xvec,uvec,xdes=None,Q=None,R=None):
    if Q is None:
        Q = np.eye(xvec.shape[0])
    if R is None:
        R = np.eye(uvec.shape[0])
    if xdes is None:
        xd = np.zeros(xvec.shape)
    elif len(xdes.shape) == 1:
        xd = np.repeat(xdes.reshape(-1,1),xvec.shape[1],axis=1)

    c = 0
    for i in range(xvec.shape[1]):
        c+=(xvec[:,i]-xd[:,i]).dot(Q).dot((xvec[:,i]-xd[:,i]).T) + uvec[:,i].dot(R).dot(uvec[:,i].T)
    return c

def rattling_objective(xvec, uvec, dt=0.05, w1=1, w2=1, coord_fun=None, w_sz=20, ov=1, xdes=None, Q=None, R=None):
    c = w1*quadratic_objective(xvec,uvec,xdes,Q,R)
    if coord_fun is None:
        r = rattling_windows(xvec.T, dt, w_sz, ov)[0]
        c += w2*np.mean(r)
    else:
        r = rattling_windows(coord_fun(xvec).T, dt, w_sz, ov)[0]
        c += w2*np.mean(r)
    return c
