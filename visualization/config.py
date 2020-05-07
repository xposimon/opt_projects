


BFGS = 2
ELS = 3
NEWTON = 1
POWELL = 4
NELDERMEAD = 5
CG = 6
PT_ERROR = 1E-6

ALL_METHODS = [NEWTON, ELS, BFGS, POWELL, NELDERMEAD, CG]
TXT = {NEWTON:"Newton-CG", BFGS:'BFGS', ELS:'Exact line search', POWELL:"Powell's method", NELDERMEAD:"Nelder-Mead", CG:"Conjugate gradient"}
COLOR = {NEWTON:"#FA08E8", BFGS:'#09F822', ELS:'#F2FC18', POWELL:"#581845", NELDERMEAD:"#34495E", CG:"#9A7D0A"}
TXT_2_ID = {}
for k, v in TXT.items():
    TXT_2_ID[v] = k
