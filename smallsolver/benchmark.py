import cvxpy as cvx
import numpy as np
import time
from project import qp

Q= [[2,0],[0,2]]
c = [[1], [4]]
A = [[1,1]]
b = [[5]]
x= [[1],[1]]

def benchmark1(Q, c, A, b, itr_rnd=1000):
    qp_start = time.time()
    for i in range(itr_rnd-1):
        # The whole process including the assignment of parameters
        Q= [[2,0,0,0],[0,4,0,0], [0,0,3,0], [0,0,0,5]]
        c = [[1], [4], [5], [9]]
        A = [[1,1,1,3], [1,0,3,0]]
        b = [[15], [7]]
        qp(Q, A, b, c)

    Q= [[2,0,0,0],[0,4,0,0], [0,0,3,0], [0,0,0,5]]
    c = [[1], [4], [5], [9]]
    A = [[1,1,1,3], [1,0,3,0]]
    b = [[15], [7]]
    print("small solver:", qp(Q, A, b, c))
    qp_end = time.time()
    print("small sovler time:", qp_end-qp_start, "s")

def benchmark2(Q, c, A, b):
    # Test the accuracy, and efficiency
    nQ = np.array(Q)
    nc = np.array(c)
    nA = np.array(A)
    nb = np.array([i[0] for i in b])
    cvx_start = time.time()     
    nx = cvx.Variable(len(Q))
    obj = cvx.Minimize((1/2)*cvx.quad_form(nx, nQ) + nc.T @ nx)
    cons = [nA@nx==nb]
    prob = cvx.Problem(obj, cons)
    prob.solve()
    cvx_end = time.time()
    
    print(prob.status, prob.value)
    #print("cvx:", nx.value)
    print("cvx time:", cvx_end-cvx_start, "s")

    qp_start = time.time()
    x = qp(Q, A, b, c)
    qp_end = time.time()
    x = np.array(x)
    res = (1/2)*cvx.quad_form(x, nQ) + nc.T @ x
    print("solver value:",res.value)
    #print("small solver:", x.tolist())
    
    print("small sovler time:", qp_end-qp_start, "s")

# Q= [[2,0,0,0],[0,4,0,0], [0,0,3,0], [0,0,0,5]]
# c = [[1], [4], [5], [9]]
# A = [[1,1,1,3], [1,0,3,0]]
# b = [[15], [7]]
# benchmark1(Q,c,A,b, itr_rnd=1)

n = 500
p = 10
seed = int(time.time())
print("seed:", seed)
np.random.seed(1588863721)
nP = np.random.randn(n, n)
nP = nP.T @ nP
for i in range(len(nP)):
    for j in range(len(nP)):
        if i ==j:
            continue
        nP[i][j] = 0 

nq = np.random.randn(n)
nA = np.random.randn(p, n)
nb = np.random.randn(p)

Q = nP.tolist()
c = nq.tolist()
A = nA.tolist()
b = nb.tolist()
c = [[j] for j in c]
b = [[i] for i in b]
#print(c, b)
benchmark2(Q, c, A, b)