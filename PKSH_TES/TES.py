import geatpy as ea
import sys
import os
import argparse
from public import glo
import numpy as np

# wyh


def argdet():
    print(sys.argv)
    if len(sys.argv) <= 44:
        args = myargs()
        return args
    else:
        print('Cannot recognize the inputs!')
        print("-i data -opt optimizer -dim dimension")
        exit()

def myargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t', default="tdcs_I_T", help='stimulation method')
    parser.add_argument('--type2', '-t2', default="ti_I", help='stimulation method2')
    parser.add_argument('--position', '-p', default='hippo', help='target location')
    parser.add_argument('--head', '-m', default='ernie', help='head model name to util')
    parser.add_argument('--gen', '-g', default=0, help='max epochs')
    parser.add_argument('--ele_number', '-en', default=75, help='max epochs')
    parser.add_argument('--name', '-n', default="SEGA_multi", help='algrithem1 name')
    parser.add_argument('--name2', '-n2', default="NSGA2_DE", help='algrithem1 name2')
    parser.add_argument('--seed', '-s', default=1, help='seed')
    parser.add_argument('--mti_number', '-mn', default='2', help='seed')
    parser.add_argument('--all_pos', '-ap', default='no', help='seed')
    #parser.add_argument('--input', '-o', default= os.path.abspath(os.path.dirname(__file__))+'/data' , help='input path')
    #parser.add_argument('--output', '-o', default= os.path.abspath(os.path.dirname(__file__)) , help='output path')
    args = parser.parse_args()
    return args

print("start1")

args = argdet()
glo.head_model = args.head
glo.type = args.type
glo.NUM_ELE = int(args.ele_number)
glo.six_pos = args.six_pos
NUM_ELE = glo.NUM_ELE # save file


glo.name = args.position
if args.position == 'hippo':
    glo.position = np.array([-31, -20, -14])
elif args.position == 'pallidum':
    glo.position = np.array([-17, 3, -1])
elif args.position == 'thalamus':
    glo.position = np.array([10, -19, 6])
elif args.position == 'sensory':
    glo.position = [41,-36,66]
elif args.position == 'dorsal':
    glo.position = [25,42,37]
elif args.position == 'v1':
    glo.position = np.array([14,-99,-3])
elif args.position == 'dlpfc':
    glo.position = np.array([-39, 34, 37])
elif args.position == 'motor':
    glo.position = np.array([47, -13, 52])
else:
    print("coordinate")
    glo.position = np.array(args.position)

if args.type == 'tdcs_I_T':
    from tdcs_problem_intensity import MyProblem
    problem = MyProblem(pooltype='Thread', num=int(args.mti_number))
elif args.type == 'tdcs_R_T':
    from tdcs_problem_ratio import MyProblem
    problem = MyProblem(pooltype='Thread', num=int(args.mti_number))
elif args.type == 'tdcs_M_T':
    from tdcs_problem_maxintensity import MyProblem
    problem = MyProblem(pooltype='Thread', num=int(args.mti_number))
elif args.type == 'autotdcs_I_T':
    from autotdcs_problem_intensity import MyProblem
    problem = MyProblem(pooltype='Thread')
elif args.type == 'autotdcs_R_T':
    from autotdcs_problem_ratio import MyProblem
    problem = MyProblem(pooltype='Thread')
elif args.type == 'autotdcs_M_T':
    from autotdcs_problem_maxintensity import MyProblem
    problem = MyProblem(pooltype='Thread')
elif args.type == 'twotdcs_I_enum':
    from twoTES_problem_enum import MyProblem
    problem = MyProblem()
# elif args.type == 'tdcs_I_T1':
#     from mti_problem_intensity1 import MyProblem
#     problem = MyProblem()
# elif args.type == 'tdcs_I_T2':
#     from mti_problem_intensity2 import MyProblem
#     problem = MyProblem()
else:
    print('ERROR: STIMULATION TYPE')
    exit(0)

# print(2)

gen = 50
if int(args.gen) != 0:
    gen = int(args.gen)

if args.name == 'GA_enum':
    eaA = ea.soea_EGA_templet
elif args.name == 'GA':
    eaA = ea.soea_EGA_templet
elif args.name == 'PKSH':
    eaA = ea.soea_multi_SEGA_templet
else:
    print('ERROR: NO -n name ')
    exit(0)

if 'PKSH' in args.name:

    Encoding = 'RI'
    NINDs = [20, 30, 40, 50]
    population = []
    for i in range(len(NINDs)):
        Field = ea.crtfld(Encoding,
                          problem.varTypes,
                          problem.ranges,
                          problem.borders)
        population.append(ea.Population(
            Encoding, Field, NINDs[i]))
    algorithm = ea.soea_multi_SEGA_templet(
        problem,
        population,
        MAXGEN=gen,
        logTras=1,
        # trappedValue=1e-6,
        maxTrappedCount=10)
else:
    algorithm = eaA(
        problem,
        ea.Population(Encoding='RI', NIND=100),
        MAXGEN=gen,  # iteration
        logTras=1,  # print log per logTras epoch ，0 means not。
        # trappedValue=1e-6,  # early stopping parameter
        maxTrappedCount=10)

if 'DE' in args.name:
    algorithm.mutOper.F = 0.5
    algorithm.recOper.XOVR = 0.2

# wyh
res = ea.optimize(algorithm,
                  seed=int(args.seed),
                  verbose=True,
                  drawing=0,
                  outputMsg=True,
                  drawLog=False,
                  saveFlag=True,
                  dirName="stage_one_" + args.type + "_" + args.name + "_" + args.seed + "_" + args.mti_number)
print(res)
# print(res['Vars'][0])
print("GA_enum_Best_ti_I: ", glo.GA_enum_Best_ti_I)
print("GA_enum_Best_ti_intensity: ", glo.GA_enum_Best_ti_intensity)
# print("GA_only2_Best_ti_intensity: ", glo.GA_only2_Best_ti_intensity)
print("stage one over!!!!")
import getRIM
num =int(args.mti_number)
_ = np.array(res['Vars'][0])
print(_)
if 'auto' in args.type:
    num = 37
    endnum = int(_[-1])
    x1 = _[0:endnum]
    x2 = _[num:num + endnum]
    x = np.concatenate((x1, x2))
    i = _[2 * num :2 * num + endnum]
    R, I, M, N2, N, MR = getRIM.get_tdcs_lfm(x, i, endnum)

    print(1)
    s = ''
    for k in range(len(x)):
        s += str(int(x[k])) + ', '
    for t in _[2 * num :2 * num + endnum]:
        s += f'{t:.3f}' + ', '
    for t in _[2 * num :2 * num + endnum]:
        s += f'{-t:.3f}' + ', '
    print(s[:-2])
    print("intensity: ", I)
    print("ratio: ", R)
    print("max intensity: ", M)
    print("> 0.2 V/m : ", N2)
    print("> 0.2 V/m target : ", N)
    print("max avoid / max target : ", MR)
elif 'enum' in args.name:
    # print(_)
    R, I, M, N2, N, MR = getRIM.get_tdcsenum_lfm(_[0:4], glo.GA_enum_Best_ti_I, 4)

    print(2)
    s = ''
    for k in _[0:4]:
        s += str(int(k)) + ', '
    for t in glo.GA_enum_Best_ti_I:
        s += f'{t:.3f}' + ', '
    print(s[:-2])
    print("intensity: ", I)
    print("ratio: ", R)
    print("max intensity: ", M)
    print("> 0.2 V/m : ", N2)
    print("> 0.2 V/m target : ", N)
    print("max avoid / max target : ", MR)
else:
    # print(_)
    R, I, M, N2, N, MR = getRIM.get_tdcs_lfm(_[0:2*num], _[2*num:], num)

    print(3)
    s = ''
    for i in range(num * 2):
        s += str(int(_[i])) + ', '
    for i in _[2*num:]:
        s += f'{i:.3f}' + ', '
    for i in _[2*num:]:
        s += f'{-i:.3f}' + ', '
    print(s[:-2])
    print("intensity: ", I)
    print("ratio: ", R)
    print("max intensity: ", M)
    print("> 0.2 V/m : ", N2)
    print("> 0.2 V/m target : ", N)
    print("max avoid / max target : ", MR)

if 'PKSH' not in args.name:
    exit(0)
print('stage two !')
# stage two
if args.type2 == 'tdcs_2':
    from tdcs_problem2_2 import MyProblem2
    problem = MyProblem2(pooltype='Thread', num=int(args.mti_number))
elif args.type2 == 'tdcs_3':
    from tdcs_problem2_3 import MyProblem2
    problem = MyProblem2(pooltype='Thread', num=int(args.mti_number))
elif args.type2 == 'tdcs_4':
    from tdcs_problem2_4 import MyProblem2
    problem = MyProblem2(pooltype='Thread', num=int(args.mti_number))
elif args.type2 == 'autotdcs_2':
    from autotdcs_problem2_2 import MyProblem2
    problem = MyProblem2(pooltype='Thread')
elif args.type2 == 'autotdcs_3':
    from autotdcs_problem2_3 import MyProblem2
    problem = MyProblem2(pooltype='Thread')
elif args.type2 == 'autotdcs_4':
    from autotdcs_problem2_4 import MyProblem2
    problem = MyProblem2(pooltype='Thread')
else:
    print('ERROR: no STIMULATION TYPE2')
    exit(0)

if args.name2 == 'GA':
    eaA = ea.moea_awGA_templet
elif args.name2 == 'NSGA3_DE':
    eaA = ea.moea_NSGA2_DE_templet
elif args.name2 == 'PKSH':
    eaA = ea.moea_NSGA3_DE_templet
elif args.name2 == 'PPS':
    eaA = ea.moea_PPS_MOEAD_DE_archive_templet
elif args.name2 == 'RVEA':
    eaA = ea.moea_RVEA_RES_templet
else:
    print('ERROR: NO -n2 name2 ')
    exit(0)

prophetPop = res['optPop']
algorithm = eaA(
    problem,
    ea.Population(Encoding='RI', NIND=100),
    prophetPop=prophetPop,  # 传入先验知识
    MAXGEN=gen,  # 最大进化代数
    logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
    maxTrappedCount=10)
# 求解
res = ea.optimize(algorithm,
                  seed=int(args.seed),
                  verbose=True,
                  drawing=0, # 绘制结果图
                  outputMsg=True, # 是否输出参数值
                  drawLog=False, # 是否单独画best_objective
                  saveFlag=True, # 是否保存到文件夹result
                  dirName="stage_two_" + args.type + "_" + args.name + "_" + args.seed + '_' + args.type2)
print(res)

num =int(args.mti_number)

list_name = args.six_pos + "_" + args.position + "_" + args.type + "_" + args.name + "_" + args.seed + '_' + args.type2 + "_" + args.mti_number + "_" + args.gen
# 保存Array对象到文件
np.save(f'A_stageTwo_Result/{list_name}.npy', res['Vars'])

_ = np.array(res['Vars'][0])
# print(_)
if 'auto' in args.type2:
    num = 37
    endnum = int(_[-1])
    x1 = _[0:endnum]
    x2 = _[num:num + endnum]
    x = np.concatenate((x1, x2))
    i = _[2 * num :2 * num + endnum]
    R, I, M, N2, N, MR = getRIM.get_tdcs_lfm(x, i, endnum)

    print(1)
    s = ''
    for k in range(len(x)):
        s += str(int(x[k])) + ', '
    for t in _[2 * num :2 * num + endnum]:
        s += f'{t:.3f}' + ', '
    for t in _[2 * num :2 * num + endnum]:
        s += f'{-t:.3f}' + ', '
    print(s[:-2])
    print("intensity: ", I)
    print("ratio: ", R)
    print("max intensity: ", M)
    print("> 0.2 V/m : ", N2)
    print("> 0.2 V/m target : ", N)
    print("max avoid / max target : ", MR)
else:
    # print(_)
    R, I, M, N2, N, MR = getRIM.get_tdcs_lfm(_[0:2*num], _[2*num:], num)

    print(3)
    s = ''
    for i in range(num * 2):
        s += str(int(_[i])) + ', '
    for i in _[2*num:]:
        s += f'{i:.3f}' + ', '
    for i in _[2*num:]:
        s += f'{-i:.3f}' + ', '
    print(s[:-2])
    print("intensity: ", I)
    print("ratio: ", R)
    print("max intensity: ", M)
    print("> 0.2 V/m : ", N2)
    print("> 0.2 V/m target : ", N)
    print("max avoid / max target : ", MR)

