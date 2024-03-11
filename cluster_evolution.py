import numpy as np


def cluster_evolution(numTx: int, N: int):
    # 参数设置
    Lambda_G = 80  # 散射簇生成率
    Lambda_R = 4  # 散射簇消亡率
    Delta_tx = 0.1  # 发射天线间隔
    Ds = 10  # 空间相关因子，取值不确定
    t = 1000e-3  # 仿真时间10ms
    Delta_t = 100e-3  # 时间间隔1ms

    # 初始时刻，所有天线上的散射簇分布
    C = [np.zeros([numTx, N])]
    C[0][0] = np.arange(1, N + 1)
    P_survival_tx = np.exp(-Lambda_R * Delta_tx / Ds)
    for i in range(1, numTx):
        idx = 0
        for j in C[0][i - 1]:
            if j != 0:
                if np.random.rand() < P_survival_tx:
                    C[0][i][idx] = j
                else:
                    C[0][i][idx] = 0
            idx += 1

    # Δt时刻后
    Pc = 0.3  # 运动散射簇的比例
    vrx = 1  # 接收端移动速度
    vct, vcr = 0, 0.5  # 散射簇相对于发，收天线的速度
    # 新增的散射簇
    delta = Pc * (vct + vcr) * Delta_t + vrx * Delta_t

    new_cluster = np.floor(Lambda_G / Lambda_R * (1 - np.exp(-Lambda_R * delta / Ds)))
    P_survival_time = np.exp(-Lambda_R * delta / Ds)

    # 时间轴上的演进
    for k in range(1, int(t / Delta_t)):
        N += int(new_cluster)
        C.append(np.zeros([numTx, N]))
        C[k][0] = np.arange(1, N + 1)
        for i in range(1, numTx):
            idx = 0
            for j in C[k][i - 1]:
                if j != 0:
                    if np.random.rand() < P_survival_time:
                        C[k][i][idx] = j
                    else:
                        C[k][i][idx] = 0
                idx += 1
    return C


if __name__ == '__main__':
    numTx = 32
    N = 50
    C = cluster_evolution(numTx, N)  # C:簇在空时域的演进
