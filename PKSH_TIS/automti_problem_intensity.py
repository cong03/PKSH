from multiprocessing import Pool as ProcessPool
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from public import glo,util
import geatpy as ea

NUM_ELE =glo.NUM_ELE
class MyProblem(ea.Problem):
    def __init__(self, pooltype='Thread'):
        name = 'MyProblem'
        M = 1
        maxormins = [1]
        num = 37
        self.var_set = np.arange(0,NUM_ELE,1)
        self.num =num
        Dim = num * 2 + num + 1
        varTypes = [1] * Dim
        varTypes[num * 2:] = [0] * (num+1)
        varTypes[-1] = 1
        # print(varTypes)
        lb = [0] * Dim
        lb[num * 2:] = [0.1] * (num+1)
        lb[-1] = 2
        ub = [(NUM_ELE-1)] * Dim  # [(NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), 2.0, 2.0]
        ub[num * 2:] = [2.0] * (num+1)
        ub[-1] = (NUM_ELE-1) / 2
        lbin = [1] * Dim
        ubin = [1] * Dim
        # print(len(varTypes))
        # print(len(lb))
        # print(len(lbin))
        # print(len(ub))
        # print(len(ubin))

        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)
        self.PoolType = pooltype
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(12)
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())
            self.pool = ProcessPool(num_cores)

    def evalVars(self, Vars):
        N = Vars.shape[0]
        # print(N)
        # exit(0)
        args = list(zip(list(range(N)), [Vars] * N, [self.num] * N))
        if self.PoolType == 'Thread':
            res = self.pool.map(subAimFunc, args)

        elif self.PoolType == 'Process':
            res = self.pool.map_async(subAimFunc, args)
            res.wait()
        Obj = np.zeros(len(res)).tolist()
        CV = np.zeros(len(res)).tolist()
        i = 0
        for _ in res:
            Obj[i] = _[0]
            CV[i] = _[1]
            i += 1
        # print(Obj)
        # print(np.array(Obj))
        # print(np.array(CV))
        return np.array(Obj), np.array(CV)

def subAimFunc(args):
    # Vars = Vars.astype(np.int32)
    var_set = np.arange(0, NUM_ELE, 1)
    i = args[0]
    Vars = args[1]
    num = args[2]
    endnum = int(Vars[i, -1])
    x1 = var_set[np.int32(Vars[i, 0:endnum])]
    x2 = var_set[np.int32(Vars[i, num:num + endnum])]
    # print(i,x1)
    # print(i,x2)
    i = Vars[i, 2 * num :2 * num + endnum]
    lst = [int(_) for _ in x1.tolist() + x2.tolist()]
    set_lst = set(lst)
    if len(set_lst) != len(lst):
        obj_i = [1000]
    else:
        obj_i = [util.mti_I(x1, x2, i, endnum)]

    CV_i = np.sum(i) - 2.0


    return obj_i, [CV_i.tolist()]
