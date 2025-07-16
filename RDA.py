import time

import numpy as np
import random as rn

# Red Deer Algorithm (RDA)

def SortArray(Matrix):
    Output = np.zeros(Matrix.shape)
    Mean = np.zeros(Matrix.shape[0])
    for i in range(Matrix.shape[0]):
        Mean[i] = np.mean(Matrix[i, :])
    index = np.argsort(Mean)
    index = index[::-1]
    for i in range(Matrix.shape[0]):
        Output[i, :] = Matrix[index[i], :]
    return Output, index


def SortPosition(Matrix, index):
    Output = np.zeros(Matrix.shape)
    for i in range(Matrix.shape[0]):
        Output[i, :] = Matrix[index[i], :]
    return Output


def Check_Bounds(s, lb, ub):
    for i in range(s.shape[0]):
        if s[i] > ub[i]:
            s[i] = ub[i]
        if s[i] < lb[i]:
            s[i] = lb[i]
    return s


def RDA(Positions, fname, xmin, xmax, MaxIter):
    N, dim = Positions.shape[0], Positions.shape[1]
    Fit = np.zeros((N, dim))
    Fnew = np.zeros((N, dim))
    V = np.zeros((N, dim))
    P = np.zeros((N, dim))
    offs = np.zeros((N, dim))
    Distance = np.zeros(N)
    N_harem = np.zeros((N, dim))

    for i in range(N):
        Fit[i, :] = fname(Positions[i, :])
        Fit[i, :] = Check_Bounds(Fit[i, :], xmin[i, :], xmax[i, :])

    ct = time.time()
    for t in range(MaxIter):
        print(t)
        num_of_com = round(0.5 * N)
        num_of_stag = N - num_of_com
        Fitness, index = SortArray(Fit)
        Solution = SortPosition(Positions, index)
        for i in range(N):
            if rn.uniform(0, 1) >= 0.5:
                Temp = Solution[i, :] + rn.uniform(0, 1) * (((xmax[i, :] - xmin[i, :]) * rn.uniform(0, 1)) + xmin[i, :])
            else:
                Temp = Solution[i, :] - rn.uniform(0, 1) * (((xmax[i, :] - xmin[i, :]) * rn.uniform(0, 1)) + xmin[i, :])
            a = np.mean(Temp)
            b = np.mean(Solution[i, :])
            if a > b:
                Solution[i, :] = Temp

        num_of_com = round(0.5 * N)
        num_of_stag = N - num_of_com
        Fitness, index = SortArray(Fit)
        Solution = SortPosition(Positions, index)

        for i in range(num_of_com):
            for j in range(num_of_stag):
                New1 = ((Solution[i, :] + Solution[num_of_com + j, :]) / 2) + rn.uniform(0, 1) * (
                            ((xmax[i, :] - xmin[i, :]) * rn.uniform(0, 1)) + xmin[i, :])
                New2 = ((Solution[i, :] + Solution[num_of_com + j, :]) / 2) + rn.uniform(0, 1) * (
                            ((xmax[i, :] - xmin[i, :]) * rn.uniform(0, 1)) + xmin[i, :])
                a = np.mean(New1)
                b = np.mean(Solution[i, :])
                if a > b:
                    Solution[i, :] = New1
                a = np.mean(New2)
                b = np.mean(Solution[num_of_com + i, :])
                if a > b:
                    Solution[num_of_com + i, :] = New2

        for i in range(N):
            Fnew[i, :] = fname(Solution[i, :])
            Fnew[i, :] = Check_Bounds(Fnew[i, :], xmin[i, :], xmax[i, :])
            V[i, :] = Fnew[i, :] - max(Fnew[i, :])
            P[i, :] = abs(V[i, :]) / sum(Fnew[i, :])
            N_harem[i, :] = np.round(P[i, :]) * index[i]

        for i in range(num_of_com):
            N_harem[i, :] = np.round(Solution[i, :] * N_harem[i, :])
            offs[i, :] = ((Solution[i, :] + index[i]) / 2) + (xmax[i, :] - xmin[i, :]) * rn.uniform(0, 1)
            k = rn.randrange(10)
            N_harem[i, :] = np.round(rn.uniform(0, 1) * N_harem[k, :])

        for i in range(num_of_stag):
            Distance[i] = np.sqrt(sum((Solution[num_of_com + i, :] - index[num_of_com + i]) ** 2))
            offs[num_of_com + i, :] = ((Solution[num_of_com + i, :] + index[num_of_com + i]) / 2) + (
                        xmax[i, :] - xmin[i, :]) * rn.uniform(0, 1)

        for i in range(N):
            a = np.mean(offs[i, :])
            b = np.mean(Solution[i, :])
            if a > b:
                Solution[i, :] = offs[i, :]
            a = np.mean(Fnew[i, :])
            b = np.mean(Fitness[i, :])
            if a > b:
                Fitness[i, :] = Fnew[i, :]

    ct = time.time() - ct
    Sol, ind = SortArray(Solution)
    return Fnew, Fitness, Sol[0, :], ct
