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
        self.var_set = np.arange(0,NUM_ELE,1)
        Dim = 5
        varTypes = [1] * Dim
        lb = [0, 0, 0, 0, 0]
        ub = [(NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1)]
        lbin = [1] * Dim
        ubin = [1] * Dim

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
        args = list(zip(list(range(N)), [Vars] * N))
        if self.PoolType == 'Thread':
            res = self.pool.map(subAimFunc, args)
            Obj = np.zeros(len(res)).tolist()
            i = 0
            for _ in res:
                Obj[i] = _
                i += 1
        elif self.PoolType == 'Process':
            result= self.pool.map_async(subAimFunc, args)
            result.wait()
            Obj = np.array(result.get())
        return np.array(Obj)
def subAimFunc(args):
    # Vars = Vars.astype(np.int32)
    var_set = np.arange(0, NUM_ELE, 1)
    i = args[0]
    Vars = args[1]
    x1 = var_set[np.int32(Vars[i, [0]])]
    x2 = var_set[np.int32(Vars[i, [1]])]
    x3 = var_set[np.int32(Vars[i, [2]])]
    x4 = var_set[np.int32(Vars[i, [3]])]
    x5 = 0
    lst = [int(x1), int(x2), int(x3), int(x4)]
    set_lst = set(lst)
    obj_i = [util.twotdcs_enum_I(x1, x2, x3, x4, x5)]
    if len(set_lst) != len(lst):
        obj_i = [1000]

    return obj_i