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
        Dim = 37 * 3 + 1
        varTypes = [1] * Dim
        varTypes[74:] = [0] * 38
        # print(varTypes)
        lb = [0] * Dim
        lb[74:] = [0.1] * 38
        ub = [(NUM_ELE-1)] * Dim  # [(NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), 2.0, 2.0]
        ub[74:] = [2.0] * 38
        ub[-1] = 1.0
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

        # exit()
        return np.array(Obj), np.array(CV)

def subAimFunc(args):
    # Vars = Vars.astype(np.int32)
    var_set = np.arange(0, NUM_ELE, 1)
    i = args[0]
    ii = i
    Vars = args[1]
    x1 = var_set[np.int32(Vars[i, 0:37])]
    x2 = var_set[np.int32(Vars[i, 37:74])]
    i1 = Vars[i, 74:-1]
    r = Vars[i, -1]

    x_idx = [False] * 75
    used_i = [False] * 37
    stimulation1 = np.zeros(75)
    stimulation2 = np.zeros(75)
    k1 = 0
    sum_I = 0
    I_list = []
    for i in range(37):
        if sum_I + i1[i] > 2.0:
            break
        if x_idx[x1[i]] or x_idx[x2[i]]:
            continue
        stimulation1[x1[i]] = i1[i] * r
        stimulation1[x2[i]] = - i1[i] * r
        I_list.append(i1[i])
        k1 += 1
        x_idx[x1[i]] = True
        x_idx[x2[i]] = True
        used_i[i] = True
        sum_I += i1[i]
        if k1 > 37 / 2:
            break

    sum_I = 0
    k = 0
    for i in range(37):
        if sum_I + i1[i] > 2.0:
            break
        if x_idx[x1[i]] or x_idx[x2[i]]:
            continue
        stimulation2[x1[i]] = I_list[k] * (1 - r)
        stimulation2[x2[i]] = - I_list[k] * (1 - r)
        k += 1
        x_idx[x1[i]] = True
        x_idx[x2[i]] = True
        used_i[i] = True
        sum_I += i1[i]
        if k >= k1:
            break

    obj_i = [util.multi_tis_R(stimulation1, stimulation2)]

    used_i = np.array(used_i)

    CV_i = np.sum(abs(i1[used_i == True])) - 2.0

    return obj_i, [CV_i.tolist()]
