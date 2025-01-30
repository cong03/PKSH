from multiprocessing import Pool as ProcessPool
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from public import glo,util
import geatpy as ea

NUM_ELE =glo.NUM_ELE
class MyProblem2(ea.Problem):
    def __init__(self, M=2, pooltype='Thread', num=4):
        name = 'MyProblem2'
        maxormins = [1] * M
        self.var_set = np.arange(0,NUM_ELE,1)
        self.num =num
        Dim = num * 2 + num
        varTypes = [1] * Dim
        varTypes[num * 2:] = [0] * num
        # print(varTypes)
        lb = [0] * Dim
        lb[num * 2:] = [0.1] * num
        ub = [(NUM_ELE-1)] * Dim  # [(NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), 2.0, 2.0]
        ub[num * 2:] = [2.0] * num
        lbin = [1] * Dim
        ubin = [1] * Dim
        # print(321)

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
        r1 = np.zeros(len(res)).tolist()
        r2 = np.zeros(len(res)).tolist()
        CV = np.zeros(len(res)).tolist()
        i = 0
        for _ in res:
            r1[i] = _[0]
            r2[i] = _[1]
            CV[i] = _[2]
            i += 1

        Obj = np.hstack([r1, r2])
        return Obj, np.array(CV)

def subAimFunc(args):
    # Vars = Vars.astype(np.int32)
    var_set = np.arange(0, NUM_ELE, 1)
    i = args[0]
    Vars = args[1]
    num = args[2]
    x = var_set[np.int32(Vars[i, 0:2 * num])]
    i = Vars[i, 2 * num :]
    lst = [int(_) for _ in x.tolist()]
    set_lst = set(lst)
    if len(set_lst) != len(lst):
        CV_i = np.sum(i) - 2.0
        return [1000], [1000], [CV_i.tolist()]
    _i, _r = util.tdcs2_two(x, i, num)

    CV_i = np.sum(i) - 2.0

    return [_i], [_r], [CV_i.tolist()]

