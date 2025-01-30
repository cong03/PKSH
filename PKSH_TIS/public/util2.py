import numpy as np
# import cupy as np
import math
from public import glo

# wyh

inf = 9999999999
NUM_ELE = glo.NUM_ELE

POINT_NUM = 1000



print("glo: ",glo,"head_model: ",glo.head_model)
if glo.head_model == 'hcp4':
    print("loading hcp4")
    # grey and white matter
    lfm = np.load(r'./data/lfm_hcp4_20.npy')
    print("load grey and white matter: ",lfm.shape)
    pos = np.load(r'./data/pos_hcp4_20.npy')
    print("load position: ", pos.shape)

if glo.head_model == 'hcpgroup':
    print("loading hcpgroup")
    # grey and white matter
    lfm = np.load(r'./data/lfm_hcpgroup.npy')
    print("load grey and white matter: ",lfm.shape)
    pos = np.load(r'./data/pos_hcpgroup.npy')
    print("load position: ", pos.shape)

if glo.head_model == 'ernie':
    print("loading ernie")
    # grey and white matter
    lfm = np.load(r'./data/lfm_ernie.npy')
    print("load grey and white matter: ",lfm.shape)
    pos = np.load(r'./data/pos_ernie.npy')
    print("load position: ", pos.shape)

#for avoidance 
# position =[-39, 34, 37] # dlpfc
# distance = np.zeros(len(pos))
# print(position)

# for i in range(len(pos)):
#     distance[i] = (pos[i, 0] - position[0])**2 + (pos[i, 1] - position[1])**2 + (pos[i, 2] - position[2])**2
# AVOID_POSITION = np.where(distance < 10**2)
# AVOID_POSITION = AVOID_POSITION[0]
# print(len(AVOID_POSITION))

position =  glo.position 
distance = np.zeros(len(pos))
print("roi_positon: ", position)

for i in range(len(pos)):
    distance[i] = (pos[i, 0] - position[0])**2 + (pos[i, 1] - position[1])**2 + (pos[i, 2] - position[2])**2

print('min_distance:' + str(min(distance)))
TARGET_POSITION = np.where(distance <= 10**2) #roi size (CM^2)
TARGET_POSITION = TARGET_POSITION[0]
AVOID_POSITION = np.where(distance > 10**2)
AVOID_POSITION = AVOID_POSITION[0]
if glo.six_pos == 'yes':
    # 储存六个位置的坐标
    positions = [
        np.array([-39, 34, 37]),
        np.array([47, -13, 52]),
        np.array([14,-99,-3]),
        np.array([-31, -20, -14]),
        np.array([10, -19, 6])
    ]

    # 计算目标（target）和避开（avoid）位置
    for position in positions:
        distance = np.zeros(len(pos))
        for i in range(len(pos)):
            distance[i] = (pos[i, 0] - position[0]) ** 2 + (pos[i, 1] - position[1]) ** 2 + (
                        pos[i, 2] - position[2]) ** 2

        print("Position:", position)
        print('min_distance:', min(distance))

        # 计算目标（target）和避开（avoid）位置
        tem_TARGET_POSITION = np.where(distance <= 10 ** 2)[0]
        tem_AVOID_POSITION = np.where(distance > 10 ** 2)[0]

        print('tem TARGET_POSITION:', len(tem_TARGET_POSITION))
        print('tem AVOID_POSITION:', len(tem_AVOID_POSITION))

        # 合并目标（target）和避开（avoid）位置
        combined_targets = np.union1d(tem_TARGET_POSITION, TARGET_POSITION)
        combined_avoids = np.union1d(tem_AVOID_POSITION, AVOID_POSITION)

        print('Combined Targets:', len(combined_targets))
        print('Combined Avoids:', len(combined_avoids))

        # 更新全局目标（target）和避开（avoid）位置
        TARGET_POSITION = combined_targets
        AVOID_POSITION = combined_avoids
        AVOID_POSITION = np.setdiff1d(AVOID_POSITION, TARGET_POSITION)

print('volume in all:' + str(len(pos)))
print('volume in avoid:' + str(len(AVOID_POSITION)))
print('volume in roi:' + str(len(TARGET_POSITION)))

# 两个电场三维度叠加（与TI开创文章提到的公式一致）
def envelop(e1,e2):
    eam = np.zeros(len(e1))
    l_x = np.sqrt(np.sum(e1 * e1, axis=1)) # 计算e1模长
    l_y = np.sqrt(np.sum(e2 * e2, axis=1)) # 计算e2模长
    l = l_x * l_y

    # wyh :Add logic to handle division by zero or invalid values
    d_zero_indices = np.where(l == 0)  # Find indices where d is zero
    d_invalid_indices = np.where(np.isnan(l))  # Find indices where d is NaN
    l[d_zero_indices] = 1e-16  # Replace zeros with 1e-9 to avoid division by zero
    l[d_invalid_indices] = 10000000.0  # Replace NaN with 10000000

    point = np.sum(e1 * e2, axis=1) # 计算点乘
    cos_ = point / l # 计算e1和e2夹角余弦值

    mask = cos_ <= 0 # 发现e1和e2夹角小于90度的点
    e1[mask] = -e1[mask]
    cos_[mask] = -cos_[mask]

    equal_vectors = np.all(e1 == e2, axis=1) # 得到e1和e2在维度2上所有值 都相等 的点
    
    eam[equal_vectors] = 2 * l_x[equal_vectors] # 1、x y z方向上的值都相等，直接等于
    not_equal_vectors = ~equal_vectors # 2、x y z方向上有不同值的情况
    mask2 = not_equal_vectors & (l_y < l_x) # 2.1、y模长小于x
    # mask3 = not_equal_vectors & (l_x < l_y * cos_) # 2.2、x模长小于y模长乘以cos值
    mask3 = not_equal_vectors & (l_y < l_x * cos_) # 2.2、y模长小于x模长乘以cos值
    # print("1:",mask3)
    # print("2:",mask4)
    # print("ly:",l_y[-3:])
    # print("lx:",l_x[-3:])
    # print("lycos",l_y[-3:] * cos_[-3:])
    # print("lxcos",l_x[-3:] * cos_[-3:])
    # if np.all(mask3 != mask4):
    #     print("no")
    #     exit(0)
    eam[mask2 & mask3] = 2 * l_y[mask2 & mask3]
    eam[mask2 & ~mask3] = 2 * np.linalg.norm(np.cross(e2[mask2 & ~mask3], (e1[mask2 & ~mask3] - e2[mask2 & ~mask3])), axis=1) / np.linalg.norm(e1[mask2 & ~mask3] - e2[mask2 & ~mask3], axis=1)

    # mask4 = not_equal_vectors & (l_y < l_x * cos_)
    mask4 = not_equal_vectors & (l_x < l_y * cos_)
    mask5 = not_equal_vectors & (l_x < l_y)
    eam[mask5 & mask4] = 2 * l_x[~mask2 & mask4]
    eam[mask5 & ~mask4] = 2 * np.linalg.norm(np.cross(e1[mask5 & ~mask4], (e2[mask5 & ~mask4] - e1[mask5 & ~mask4])), axis=1) / np.linalg.norm(e2[mask5 & ~mask4] - e1[mask5 & ~mask4], axis=1)
   
    return eam

# for 两对TIS
# max intensity for stage one
# stage one： ti遗传算法的目标式子，最大化目标电场强度 1 -1 1 -1
def tis_function5(x1, x2, x3, x4,x5):

    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = 1 + x5/NUM_ELE
    stimulation1[electrode2] = -1 - x5/NUM_ELE
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000 # 计算75个电极三个方向上产生的电场强度分量值

    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = 1 - x5/NUM_ELE
    stimulation2[electrode4] = -1 + x5/NUM_ELE
    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    eam = envelop(e1,e2)
    return 1/np.mean(abs(eam)) 



# GA+枚举搜索 固定2mA最优的4电极位置和电极配比（目标函数：靶区最大平均电场强度）
def tis_function5_enum_I(x1, x2, x3, x4,x5):
    maxeam = 0
    i_ = 0
    for i in np.arange(0.1, 2, 0.05):
        if float(i) < x5/NUM_ELE:
            continue
        electrode1 = x1
        electrode2 = x2
        stimulation1 = np.zeros(NUM_ELE)
        stimulation1[electrode1] = float(i) + x5/NUM_ELE
        stimulation1[electrode2] = -float(i) - x5/NUM_ELE
        e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000

        electrode3 = x3
        electrode4 = x4
        stimulation2 = np.zeros(NUM_ELE)

        stimulation2[electrode3] = float(2.0 - i) - x5/NUM_ELE
        stimulation2[electrode4] = -float(2.0 - i) + x5/NUM_ELE
        e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
        eam = envelop(e1,e2)
        if maxeam < np.mean(abs(eam)):
            maxeam = np.mean(abs(eam))
            i_ = i

    if 1/maxeam < glo.GA_enum_Best_ti_intensity:
        glo.GA_enum_Best_ti_intensity = 1/maxeam
        glo.GA_enum_Best_ti_I = np.array([i_, -i_, 2.0-i_, i_-2.0])
    return 1/maxeam

# GA+枚举搜索 固定2mA最优的4电极位置和电极配比（目标函数：R）
def tis_function5_enum_R(x1, x2, x3, x4,x5):
    mineam = np.float64(999.0)
    i_ = 0.0
    opt = np.float64(999.0)
    for i in np.arange(0.1, 2, 0.05):
        if float(i) < x5/NUM_ELE:
            continue
        electrode1 = x1
        electrode2 = x2
        stimulation1 = np.zeros(NUM_ELE)
        stimulation1[electrode1] = float(i) + x5 / NUM_ELE
        stimulation1[electrode2] = -float(i) - x5 / NUM_ELE
        e1 = np.array(
            [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
             np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
             np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T / 1000

        electrode3 = x3
        electrode4 = x4
        stimulation2 = np.zeros(NUM_ELE)

        stimulation2[electrode3] = float(2.0 - i) - x5 / NUM_ELE
        stimulation2[electrode4] = -float(2.0 - i) + x5 / NUM_ELE
        e2 = np.array(
            [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
             np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
             np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
        eam_target = envelop(e1, e2)  # shape(x,)
        e1_ = np.array(
            [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
             np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T / 1000
        e2_ = np.array(
            [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
             np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
        eam_avoid = envelop(e1_, e2_)  # shape(x,)
        # print(type(mineam))
        # print(type(np.mean(abs(eam_avoid)) / np.mean(abs(eam_target))))
        # exit(0)
        # print(mineam)
        # print(np.mean(abs(eam_avoid)) / np.mean(abs(eam_target)))
        mineam = np.mean(abs(eam_avoid)) / np.mean(abs(eam_target))
        if mineam < opt:
            opt = mineam
            i_ = i
        # exit(0)
    # print(opt)
    # exit(0)
    if opt < glo.GA_enum_Best_ti_intensity:
        glo.GA_enum_Best_ti_intensity = opt
        glo.GA_enum_Best_ti_I = np.array([i_, -i_, 2.0-i_, i_-2.0])
    return opt

# GA+枚举搜索 固定2mA最优的4电极位置和电极配比（目标函数：M）
def tis_function5_enum_M(x1, x2, x3, x4,x5):
    maxeam = 0
    i_ = 0
    for i in np.arange(0.1, 2, 0.05):
        if float(i) < x5/NUM_ELE:
            continue
        electrode1 = x1
        electrode2 = x2
        stimulation1 = np.zeros(NUM_ELE)
        stimulation1[electrode1] = float(i) + x5/NUM_ELE
        stimulation1[electrode2] = -float(i) - x5/NUM_ELE
        e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000

        electrode3 = x3
        electrode4 = x4
        stimulation2 = np.zeros(NUM_ELE)

        stimulation2[electrode3] = float(2.0 - i) - x5/NUM_ELE
        stimulation2[electrode4] = -float(2.0 - i) + x5/NUM_ELE
        e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
        eam = envelop(e1,e2)
        if maxeam < np.max(abs(eam)):
            maxeam = np.max(abs(eam))
            i_ = i

    if 1/maxeam < glo.GA_enum_Best_ti_intensity:
        glo.GA_enum_Best_ti_intensity = 1/maxeam
        glo.GA_enum_Best_ti_I = np.array([i_, -i_, 2.0-i_, i_-2.0])
    return 1/maxeam

# GA搜索最优的4电极位置和电极配比（目标函数：最大化 靶区平均电场强度）
def tis_function5_only_intensity(x1, x2, x3, x4,x5, i1, i2):
    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = i1 + x5 / NUM_ELE
    stimulation1[electrode2] = -i1 - x5 / NUM_ELE
    e1 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T / 1000

    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = i2 - x5 / NUM_ELE
    stimulation2[electrode4] = -i2 + x5 / NUM_ELE
    e2 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    eam = envelop(e1, e2)
    return 1/np.mean(abs(eam))

# GA搜索最优的4电极位置和电极配比（目标函数：最小化 非靶区平均电场强度/靶区平均电场强度）+ 损失函数
def tis_function5_only_ratio_p(x1, x2, x3, x4,x5, i1, i2):
    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = i1 + x5 / NUM_ELE
    stimulation1[electrode2] = -i1 - x5 / NUM_ELE
    e1 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T / 1000

    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = i2 - x5 / NUM_ELE
    stimulation2[electrode4] = -i2 + x5 / NUM_ELE
    e2 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    eam_target = envelop(e1, e2) # shape(x,)
    e1_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T / 1000
    e2_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
    eam_avoid = envelop(e1_, e2_) # shape(x,)
    et_avg = np.mean(abs(eam_target))
    # if et_avg > glo.GA_only2_Best_ti_intensity:
    #     glo.GA_only2_Best_ti_intensity = et_avg
    is_in_target = True
    # 判断 最大平均电厂强度 是否在靶区
    if np.max(abs(eam_target)) < np.max(abs(eam_avoid)):
        # print(np.max(abs(eam_target)))
        # print(np.max(abs(eam_avoid)))
        is_in_target = False
        # exit(0)
    return np.mean(abs(eam_avoid))/et_avg, is_in_target

# GA搜索最优的4电极位置和电极配比（目标函数：最小化 非靶区平均电场强度/靶区平均电场强度）
def tis_function5_only_ratio(x1, x2, x3, x4,x5, i1, i2):
    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = i1 + x5 / NUM_ELE
    stimulation1[electrode2] = -i1 - x5 / NUM_ELE
    e1 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T / 1000

    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = i2 - x5 / NUM_ELE
    stimulation2[electrode4] = -i2 + x5 / NUM_ELE
    e2 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    eam_target = envelop(e1, e2) # shape(x,)
    e1_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T / 1000
    e2_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
    eam_avoid = envelop(e1_, e2_) # shape(x,)
    et_avg = np.mean(abs(eam_target))
    # if et_avg > glo.GA_only2_Best_ti_intensity:
    #     glo.GA_only2_Best_ti_intensity = et_avg
    return np.mean(abs(eam_avoid))/et_avg, True

# GA搜索最优的4电极位置和电极配比（目标函数：最大化 靶区最大电场强度）+ 损失函数
def tis_function5_only_maxintensity_p(x1, x2, x3, x4,x5, i1, i2):
    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = i1 + x5 / NUM_ELE
    stimulation1[electrode2] = -i1 - x5 / NUM_ELE
    e1 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T / 1000

    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = i2 - x5 / NUM_ELE
    stimulation2[electrode4] = -i2 + x5 / NUM_ELE
    e2 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    eam_target = envelop(e1, e2) # shape(x,)
    e1_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T / 1000
    e2_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
    eam_avoid = envelop(e1_, e2_) # shape(x,)
    et_max = np.max(np.max(abs(eam_target)))

    # 判断 电场强度的最大值 是否在靶区；不在则对其函数值进行惩罚(f1[idx] + np.max(f1) - np.min(f1))
    is_in_target = True
    if et_max < np.max(abs(eam_avoid)):
        is_in_target = False
    return 1/et_max, is_in_target

# GA搜索最优的4电极位置和电极配比（目标函数：最大化 靶区最大电场强度）
def tis_function5_only_maxintensity(x1, x2, x3, x4,x5, i1, i2):
    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = i1 + x5 / NUM_ELE
    stimulation1[electrode2] = -i1 - x5 / NUM_ELE
    e1 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T / 1000

    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = i2 - x5 / NUM_ELE
    stimulation2[electrode4] = -i2 + x5 / NUM_ELE
    e2 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    eam_target = envelop(e1, e2) # shape(x,)

    et_max = np.max(abs(eam_target))

    # 判断 电场强度的最大值 是否在靶区；不在则对其函数值进行惩罚(f1[idx] + np.max(f1) - np.min(f1))
    is_in_target = True

    return 1/et_max, is_in_target

# stage two: 第二步骤，三目标优化（最大化 靶区平均电场强度、最小化 非靶区平均电场强度/靶区平均电场强度、最大化 靶区最大电场强度）
def tis2_two(x1, x2, x3, x4,x5, i1, i2):
    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = i1 + x5 / NUM_ELE
    stimulation1[electrode2] = -i1 - x5 / NUM_ELE
    e1 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T / 1000

    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = i2 - x5 / NUM_ELE
    stimulation2[electrode4] = -i2 + x5 / NUM_ELE
    e2 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    eam_target = envelop(e1, e2) # shape(x,)
    e1_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T / 1000
    e2_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
    eam_avoid = envelop(e1_, e2_) # shape(x,)
    is_in_target = True  # 判断 电场强度的最大值 是否在靶区；不在则对其函数值进行惩罚(f1[idx] + np.max(f1) - np.min(f1))
    et_avg = np.mean(abs(eam_target))
    et_max = np.max(abs(eam_target))
    # 判断 最大平均电厂强度 是否在靶区
    # if et_max < np.max(abs(eam_avoid)):
    #     is_in_target = False
    # 最大化 靶区平均电场强度、最小化 非靶区平均电场强度/靶区平均电场强度、最大化 靶区最大电场强度
    return 1/et_avg, np.mean(abs(eam_avoid))/et_avg, is_in_target


# stage two: 第二步骤，三目标优化（最大化 靶区平均电场强度、最小化 非靶区平均电场强度/靶区平均电场强度、最大化 靶区最大电场强度）
def tis2_three(x1, x2, x3, x4,x5, i1, i2):
    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = i1 + x5 / NUM_ELE
    stimulation1[electrode2] = -i1 - x5 / NUM_ELE
    e1 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T / 1000

    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = i2 - x5 / NUM_ELE
    stimulation2[electrode4] = -i2 + x5 / NUM_ELE
    e2 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    eam_target = envelop(e1, e2) # shape(x,)
    e1_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T / 1000
    e2_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
    eam_avoid = envelop(e1_, e2_) # shape(x,)
    is_in_target = True  # 判断 电场强度的最大值 是否在靶区；不在则对其函数值进行惩罚(f1[idx] + np.max(f1) - np.min(f1))
    et_avg = np.mean(abs(eam_target))
    et_max = np.max(abs(eam_target))
    # 判断 最大平均电厂强度 是否在靶区
    # if et_max < np.max(abs(eam_avoid)):
    #     is_in_target = False
    # 最大化 靶区平均电场强度、最小化 非靶区平均电场强度/靶区平均电场强度、最大化 靶区最大电场强度
    return 1/et_avg, np.mean(abs(eam_avoid))/et_avg, 1/et_max, is_in_target

# stage two: 第二步骤，三目标优化（最大化 靶区平均电场强度、最小化 非靶区平均电场强度/靶区平均电场强度、最大化 靶区最大电场强度）
def tis2_four(x1, x2, x3, x4,x5, i1, i2):
    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = i1 + x5 / NUM_ELE
    stimulation1[electrode2] = -i1 - x5 / NUM_ELE
    e1 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T / 1000

    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = i2 - x5 / NUM_ELE
    stimulation2[electrode4] = -i2 + x5 / NUM_ELE
    e2 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    eam_target = envelop(e1, e2) # shape(x,)
    e1_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T / 1000
    e2_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
    eam_avoid = envelop(e1_, e2_) # shape(x,)
    is_in_target = True  # 判断 电场强度的最大值 是否在靶区；不在则对其函数值进行惩罚(f1[idx] + np.max(f1) - np.min(f1))
    et_avg = np.mean(abs(eam_target))
    et_max = np.max(abs(eam_target))
    # 判断 最大平均电厂强度 是否在靶区
    # if et_max < np.max(abs(eam_avoid)):
    #     is_in_target = False
    # 最大化 靶区平均电场强度、最小化 非靶区平均电场强度/靶区平均电场强度、最大化 靶区最大电场强度
    return 1/et_avg, np.mean(abs(eam_avoid))/et_avg, 1/et_max, is_in_target, np.max(abs(eam_avoid))/et_max

def mti_I(x1, x2, i, num):
    # x = np.array(x)
    # x[abs(x[:]) < 0.01] = 0
    # return 1000 / np.average(((np.matmul(lfm[:, TARGET_POSITION, 0].T, x)) ** 2 + (
    #     np.matmul(lfm[:, TARGET_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 2].T, x)) ** 2) ** 0.5)
    stimulation1 = np.zeros(NUM_ELE)
    stimulation2 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num/2)):
        stimulation1[x1[k]] = i[k]
        stimulation1[x2[k]] = -i[k]
    for k in range(int(num/2)):
        stimulation2[x1[int(num/2) + k]] = i[int(num/2) + k]
        stimulation2[x2[int(num/2) + k]] = -i[int(num/2) + k]
    # stimulation1[x1[1]] = i[1]
    # stimulation1[x1[2]] = i[2]
    # stimulation1[x1[3]] = i[3]
    # stimulation2[x2[1]] = -i[1]
    # stimulation2[x2[2]] = -i[2]
    # stimulation2[x2[3]] = -i[3]
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000 # 计算75个电极三个方向上产生的电场强度分量值

    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000

    eam = envelop(e1,e2)

    return 1/np.mean(abs(eam))

def mti_R(x1, x2, i, num):
    # x = np.array(x)
    # x[abs(x[:]) < 0.01] = 0
    # return 1000 / np.average(((np.matmul(lfm[:, TARGET_POSITION, 0].T, x)) ** 2 + (
    #     np.matmul(lfm[:, TARGET_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 2].T, x)) ** 2) ** 0.5)
    stimulation1 = np.zeros(NUM_ELE)
    stimulation2 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num/2)):
        stimulation1[x1[k]] = i[k]
        stimulation1[x2[k]] = -i[k]
    for k in range(int(num/2)):
        stimulation2[x1[int(num/2) + k]] = i[int(num/2) + k]
        stimulation2[x2[int(num/2) + k]] = -i[int(num/2) + k]
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000 # 计算75个电极三个方向上产生的电场强度分量值

    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000

    eam_target = envelop(e1,e2)

    e1_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T /1000 # 计算75个电极三个方向上产生的电场强度分量值

    e2_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000

    eam_avoid = envelop(e1_,e2_)
    return np.mean(abs(eam_avoid))/np.mean(abs(eam_target))

def mti_M(x1, x2, i, num):
    # x = np.array(x)
    # x[abs(x[:]) < 0.01] = 0
    # return 1000 / np.average(((np.matmul(lfm[:, TARGET_POSITION, 0].T, x)) ** 2 + (
    #     np.matmul(lfm[:, TARGET_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 2].T, x)) ** 2) ** 0.5)
    stimulation1 = np.zeros(NUM_ELE)
    stimulation2 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num/2)):
        stimulation1[x1[k]] = i[k]
        stimulation1[x2[k]] = -i[k]
    for k in range(int(num/2)):
        stimulation2[x1[int(num/2) + k]] = i[int(num/2) + k]
        stimulation2[x2[int(num/2) + k]] = -i[int(num/2) + k]
    # stimulation1[x1[1]] = i[1]
    # stimulation1[x1[2]] = i[2]
    # stimulation1[x1[3]] = i[3]
    # stimulation2[x2[1]] = -i[1]
    # stimulation2[x2[2]] = -i[2]
    # stimulation2[x2[3]] = -i[3]
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000 # 计算75个电极三个方向上产生的电场强度分量值

    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000

    eam = envelop(e1,e2)

    return 1/np.max(abs(eam))

def multi_tis_I(stimulation1, stimulation2):
    # print(np.sum(abs(stimulation2)))
    # print(np.sum(abs(stimulation1)))
    if np.sum(abs(stimulation2)) < 0.001 or np.sum(abs(stimulation1)) < 0.001:
        return 1000
    # x = np.array(x)
    # x[abs(x[:]) < 0.01] = 0
    # return 1000 / np.average(((np.matmul(lfm[:, TARGET_POSITION, 0].T, x)) ** 2 + (
    #     np.matmul(lfm[:, TARGET_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 2].T, x)) ** 2) ** 0.5)
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000 # 计算75个电极三个方向上产生的电场强度分量值

    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000

    eam = envelop(e1,e2)

    return 1/np.mean(abs(eam))

def multi_tis_R(stimulation1, stimulation2):
    # print(np.sum(abs(stimulation2)))
    # print(np.sum(abs(stimulation1)))
    if np.sum(abs(stimulation2)) < 0.001 or np.sum(abs(stimulation1)) < 0.001:
        return 1000
    # x = np.array(x)
    # x[abs(x[:]) < 0.01] = 0
    # return 1000 / np.average(((np.matmul(lfm[:, TARGET_POSITION, 0].T, x)) ** 2 + (
    #     np.matmul(lfm[:, TARGET_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 2].T, x)) ** 2) ** 0.5)
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000 # 计算75个电极三个方向上产生的电场强度分量值

    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000

    eam_target = envelop(e1,e2)

    e1_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T /1000 # 计算75个电极三个方向上产生的电场强度分量值

    e2_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
    eam_avoid = envelop(e1_,e2_)


    return np.mean(abs(eam_avoid))/np.mean(abs(eam_target))


def mti2_two(x1, x2, i, num):
    # x = np.array(x)
    # x[abs(x[:]) < 0.01] = 0
    # return 1000 / np.average(((np.matmul(lfm[:, TARGET_POSITION, 0].T, x)) ** 2 + (
    #     np.matmul(lfm[:, TARGET_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 2].T, x)) ** 2) ** 0.5)
    stimulation1 = np.zeros(NUM_ELE)
    stimulation2 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num / 2)):
        stimulation1[x1[k]] = i[k]
        stimulation1[x2[k]] = -i[k]
    for k in range(int(num / 2)):
        stimulation2[x1[int(num / 2) + k]] = i[int(num / 2) + k]
        stimulation2[x2[int(num / 2) + k]] = -i[int(num / 2) + k]
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T / 1000  # 计算75个电极三个方向上产生的电场强度分量值

    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000

    eam_target = envelop(e1, e2)

    e1_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1),
                    np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
                    np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T / 1000  # 计算75个电极三个方向上产生的电场强度分量值

    e2_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2),
                    np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
                    np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000

    eam_avoid = envelop(e1_, e2_)

    eam1_avg = np.mean(abs(eam_target))
    return 1 / eam1_avg, np.mean(abs(eam_avoid)) / eam1_avg

def mti2_three(x1, x2, i, num):
    # x = np.array(x)
    # x[abs(x[:]) < 0.01] = 0
    # return 1000 / np.average(((np.matmul(lfm[:, TARGET_POSITION, 0].T, x)) ** 2 + (
    #     np.matmul(lfm[:, TARGET_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 2].T, x)) ** 2) ** 0.5)
    stimulation1 = np.zeros(NUM_ELE)
    stimulation2 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num/2)):
        stimulation1[x1[k]] = i[k]
        stimulation1[x2[k]] = -i[k]
    for k in range(int(num/2)):
        stimulation2[x1[int(num/2) + k]] = i[int(num/2) + k]
        stimulation2[x2[int(num/2) + k]] = -i[int(num/2) + k]
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000 # 计算75个电极三个方向上产生的电场强度分量值

    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000

    eam_target = envelop(e1,e2)

    e1_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T /1000 # 计算75个电极三个方向上产生的电场强度分量值

    e2_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000

    eam_avoid = envelop(e1_, e2_)

    eam1_avg = np.mean(abs(eam_target))
    eam1_max = np.max(abs(eam_target))
    return 1 / eam1_avg, np.mean(abs(eam_avoid)) / eam1_avg, 1 / eam1_max

def mti2_four(x1, x2, i, num):
    # x = np.array(x)
    # x[abs(x[:]) < 0.01] = 0
    # return 1000 / np.average(((np.matmul(lfm[:, TARGET_POSITION, 0].T, x)) ** 2 + (
    #     np.matmul(lfm[:, TARGET_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 2].T, x)) ** 2) ** 0.5)
    stimulation1 = np.zeros(NUM_ELE)
    stimulation2 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num/2)):
        stimulation1[x1[k]] = i[k]
        stimulation1[x2[k]] = -i[k]
    for k in range(int(num/2)):
        stimulation2[x1[int(num/2) + k]] = i[int(num/2) + k]
        stimulation2[x2[int(num/2) + k]] = -i[int(num/2) + k]
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000 # 计算75个电极三个方向上产生的电场强度分量值

    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000

    eam_target = envelop(e1,e2)

    e1_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T /1000 # 计算75个电极三个方向上产生的电场强度分量值

    e2_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000

    eam_avoid = envelop(e1_, e2_)

    eam1_avg = np.mean(abs(eam_target))
    eam1_max = np.max(abs(eam_target))
    return 1 / eam1_avg, np.mean(abs(eam_avoid)) / eam1_avg, 1 / eam1_max, np.max(abs(eam_avoid)) / eam1_max


def mti(x):
        x = x/2
        x[np.where(abs(x)<0.01)] = 0
        e1 = np.array([np.matmul(lfm[:, :, 0].T, x[:NUM_ELE]), np.matmul(lfm[:, :, 1].T, x[:NUM_ELE]),np.matmul(lfm[:, :, 2].T, x[:NUM_ELE])]).T /1000
        e2 = np.array([np.matmul(lfm[:, :, 0].T, x[NUM_ELE:]), np.matmul(lfm[:, :, 1].T, x[NUM_ELE:]),np.matmul(lfm[:, :, 2].T, x[NUM_ELE:])]).T /1000
        eam = envelop(e1,e2)
        return np.array([1/np.mean(eam[TARGET_POSITION]),np.mean(eam)])

def mti_avoid(x):
        x = x/2
        x[np.where(abs(x)<0.01)] = 0
        e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, x[:NUM_ELE]), np.matmul(lfm[:, TARGET_POSITION, 1].T, x[:NUM_ELE]),np.matmul(lfm[:, TARGET_POSITION, 2].T, x[:NUM_ELE])]).T /1000
        e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, x[NUM_ELE:]), np.matmul(lfm[:, TARGET_POSITION, 1].T, x[NUM_ELE:]),np.matmul(lfm[:, TARGET_POSITION, 2].T, x[NUM_ELE:])]).T /1000
        field1 = envelop(e1,e2)
        e1 = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, x[:NUM_ELE]), np.matmul(lfm[:, AVOID_POSITION, 1].T, x[:NUM_ELE]),np.matmul(lfm[:, AVOID_POSITION, 2].T, x[:NUM_ELE])]).T /1000
        e2 = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, x[NUM_ELE:]), np.matmul(lfm[:, AVOID_POSITION, 1].T, x[NUM_ELE:]),np.matmul(lfm[:, AVOID_POSITION, 2].T, x[NUM_ELE:])]).T /1000
        field2 = envelop(e1,e2)
        return np.array([1/np.mean(field1),np.mean(field2)])
