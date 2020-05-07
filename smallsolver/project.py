R,L=lambda X:range(X),lambda Y:len(Y);a,T,M=lambda A,B,C=1:[[A[i][j]+B[i][j]*C for j in R(L(A[0]))]for i in R(L(A))],lambda m:list(map(list,zip(*m))),lambda v,I,J:[r[:J]+r[J+1:]for r in (v[:I]+v[I+1:])] 
def m(a,b):n,k,m=L(a),L(b),L(b[0]);c=[[0 for i in R(m)]for j in R(n)];return[[sum([a[i][l]*b[l][j]for l in R(k)])for j in R(m)]for i in R(n)]
D=lambda m:m[0][0]*m[1][1]-m[0][1]*m[1][0]if L(m)<3 else sum([((-1)**c)*m[0][c]*D(M(m,0,c))for c in R(L(m))])
def I(m):return[[1/m[i][j] if m[i][j] else 0 for j in R(L(m))]for i in R(L(m))]
def s(A,B):
    _=list(R(L(A)))
    for f in _:
        t=1/A[f][f];B[f][0]*=t
        for j in _:A[f][j]*=t
        for i in _[0:f]+_[f+1:]: 
            u=A[i][f];B[i][0]-=u*B[f][0]
            for j in _:A[i][j]-=u*A[f][j]
    return B
def qp(Q,A,b,c):
    x=[[0]for _ in R(L(Q))];return a(x,s(Q,a(m(T(A),s(m(m(A,I(Q)),T(A)),a(m(m(A,I(Q)),c),b))),a(c,m(Q,x)),C=-1)))


# def I(m):
#     d,_=D(m),R(L(m))
#     if L(m)<3:return[[m[1][1]/d,-m[0][1]/d],[-m[1][0]/d,m[0][0]/d]]
#     h=T([[(-1)**(r+c)*D(M(m,r,c))for c in _]for r in _])
#     return[[h[r][c]/d for c in _]for r in _]