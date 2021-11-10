
import gurobipy  as gp 
from gurobipy import GRB
import numpy as np
from scipy.spatial import distance
import scipy.sparse as sp

def build_initial_constr(N, M):

    d = N*(M+1)
    A = np.zeros((M, d), 'int')
    Aeq = np.zeros((N,d), 'int')
    b = np.ones(M, 'int')
    beq = np.ones(N, 'int')

    for i in range(N):
        
        # Top
        A[:, i*(M+1):(i+1)*(M+1)] = np.hstack((np.zeros((M,1)),np.eye(M,M)))
        
        # Bottom
        Aeq[i,i*(M+1):(i+1)*(M+1)] = np.ones((1,M+1))
        
    return A, Aeq, b, beq

def build_C(in_vals, out_vals, rad):

    N = in_vals.shape[0]

    C = distance.cdist(in_vals,out_vals)
    missed = rad*np.ones((N,1))
    C_aug = np.hstack((missed,C)).flatten()
    
    return C_aug

def build_initial_all(in_vals, out_vals, rad):

    N = in_vals.shape[0]
    M = out_vals.shape[0]
    d = N*(M+1)

    C = distance.cdist(in_vals,out_vals)
    missed = rad*np.ones((N,1))
    C_aug = np.hstack((missed,C)).flatten()

    A = np.zeros((M, d), 'int')
    Aeq = np.zeros((N,d), 'int')
    b = np.ones(M, 'int')
    beq = np.ones(N, 'int')

    for i in range(N):
        
        # Top
        this_track = in_vals[i].reshape(1,3)
        det_dists = distance.cdist(this_track, out_vals)
        valid_asgns = (det_dists <= missed).astype('int')

        in_mat = np.eye(M,M)
        np.fill_diagonal(in_mat, valid_asgns)

        A[:, i*(M+1):(i+1)*(M+1)] = np.hstack((np.zeros((M,1)),in_mat))
        
        # Bottom
        Aeq[i,i*(M+1):(i+1)*(M+1)] = np.ones((1,M+1))

    return C_aug, A, Aeq, b, beq

def gurobi_ilp(f, A, b, Aeq, beq):

    try:

        # Define model
        model = gp.Model('matrix1')
        model.Params.LogToConsole = 0

        d = f.shape[0]

        # Add vars
        var = model.addMVar(shape=d, vtype=GRB.BINARY , name="x")
        obj = np.ones(d,'int')
        model.setObjective(f @ var, GRB.MINIMIZE)

        A_sp = sp.csr_matrix(A)
        Aeq_sp = sp.csr_matrix(Aeq)

        model.addConstr(A_sp @ var <= b)
        model.addConstr(Aeq_sp @ var == beq)

        model.optimize()

        x = var.X.astype('int')
        v = model.ObjVal

    except Exception as e:

        print(e)

        x = []
        v = []

    return x, v

def CalcNextBestSolution(ce, xstar, f, A, b, Aeq, beq):

    xdim = xstar.shape[0]
    A = np.vstack((A, xstar))
    b = np.hstack((b, xstar.sum() - 1))
    dd = f.shape[0]

    y = np.zeros(dd, 'int')
    assigned = np.zeros(dd, 'int')

    v = 0

    if np.size(ce) != 0:

        y[ce[:,0]] = ce[:,1]
        assigned[ce[:,0]] = 1

        b = b - A.dot(y)
        beq = beq - Aeq.dot(y)
        v = f.dot(y)

        f = np.delete(f, ce[:,0])
        A = np.delete(A, ce[:,0], axis=1)
        Aeq = np.delete(Aeq, ce[:,0], axis=1)

    y1, v1 = gurobi_ilp(f, A, b, Aeq, beq)

    if np.size(y1) == 0:
        v = 1e20

    else:

        y[assigned==0] = y1
        v = v + v1

    return v, y

def MBest(f, A, b, Aeq, beq, K):

    dd = f.shape[0]

    x = np.zeros((dd,K), 'int')
    y = np.zeros((dd,K), 'int')

    xv = np.zeros(K)
    yv = np.zeros(K)

    Constraints = {i:np.array([],dtype='int').reshape(0,2) for i in range(K)}

    for m in range(K):

        if m == 0:

            cx, value = gurobi_ilp(f, A, b, Aeq, beq)
            x[:,m] = cx
            xv[m] = value

        else:

            c = np.min(yv[:m])
            k = np.argmin(yv[:m])

            if c == 1e20:

                m = m-1
                break

            x[:,m] = y[:,k]
            xv[m] = yv[k]

            diff1 = x[:,m] != x[:,k]
            diff = np.argwhere(diff1==1)[:,0]

            Constraints[m] = np.vstack((Constraints[k], np.array([diff[0], x[diff[0], m]]).reshape(1,2)))
            Constraints[k] = np.vstack((Constraints[k], np.array([diff[0], 1-x[diff[0], m]]).reshape(1,2)))

            value, cy = CalcNextBestSolution(Constraints[k], x[:,k], f, A, b, Aeq, beq)
            y[:,k] = cy
            yv[k] = value

        value, cy = CalcNextBestSolution(Constraints[m], x[:,m], f, A, b, Aeq, beq)
        y[:,m] = cy
        yv[m] = value

    M = m
    xv = xv[:K]
    x = x[:,:K]

    return x, xv
