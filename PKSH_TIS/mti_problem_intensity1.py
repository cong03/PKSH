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
        Dim = 37 * 2
        varTypes = [1] * Dim
        varTypes[37:] = [0] * 37
        # print(varTypes)
        lb = [0] * Dim
        lb[37:] = [0.1] * 37
        ub = [(NUM_ELE-1)] * Dim  # [(NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), 2.0, 2.0]
        ub[37:] = [2.0] * 37
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

        return np.array(Obj), np.array(CV)

def subAimFunc(args):
    # Vars = Vars.astype(np.int32)
    var_set = np.arange(0, NUM_ELE, 1)
    i = args[0]
    ii = i
    Vars = args[1]
    X = var_set[np.int32(Vars[i, 0:37])]
    # print(type(x))
    I = Vars[i, 37:]
    l_X = len(X)
    x_nonull = [False] * 75
    stimulation1 = np.zeros(75)
    n_s1 = 0
    stimulation2 = np.zeros(75)
    n_s2 = 0
    used_i = [False] * 37
    sum_I = 0
    I_list = []
    for i in range(l_X):
        if i < l_X / 2:
            if x_nonull[i] or sum_I + I[i] > 2.0:
                continue
            stimulation1[i] = I[i]
            I_list.append(I[i])
            sum_I += I[i]
            x_nonull[i] = True
            used_i[i] = True
            n_s1 += 1
        else:
            if x_nonull[i] or sum_I + I[i] > 2.0:
                continue
            stimulation2[i] = I[i]
            I_list.append(I[i])
            sum_I += I[i]
            x_nonull[i] = True
            used_i[i] = True
            n_s2 += 1

    i = 0
    # print(sum_I)
    sum_I = 0
    # print(ii,I_list)
    for k in range(75):
        if len(I_list) == 0:
            break
        # print(I_list[i])
        # print(i)
        if x_nonull[k] or sum_I + I_list[i] > 2.0:
            continue

        if i < n_s1:
            stimulation1[k] = -I_list[i]
            sum_I += I_list[i]
            x_nonull[k] = True
            i += 1
        if i >= n_s1 and i < n_s1 + n_s2:
            stimulation2[k] = -I_list[i]
            sum_I += I_list[i]
            x_nonull[k] = True
            i += 1
        if i >= n_s1 + n_s2:
            break

    obj_i = [util.multi_tis_I(stimulation1, stimulation2)]
    # print(x_nonull)
    used_i = np.array(used_i)
    CV_i = np.sum(I[used_i == True]) - 2.0

    return obj_i, [CV_i.tolist()]
