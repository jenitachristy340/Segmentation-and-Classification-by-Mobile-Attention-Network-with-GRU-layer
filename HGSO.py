import time
import numpy as np


def Create_Groups(var_n_gases, var_n_types, X):
    N = var_n_gases / var_n_types
    i = 1
    Group=[]
    for j in np.arange(1, var_n_types + 1).reshape(-1):
        Group[j].Position = X[np.arange(i, i + N + 1),:]
        i = j * N + 1
        if i + N > var_n_gases:
            i = j * N

    return Group


def update_positions(Group, best_pos, vec_Xbest, S, var_n_gases, var_n_types,var_Gbest, alpha, beta, var_nvars):
    vec_flag = np.array([1, - 1])
    Group= []
    for i in np.arange(1, var_n_types + 1).reshape(-1):
        for j in np.arange(1, var_n_gases / var_n_types + 1).reshape(-1):
            gama = beta * np.exp(- (var_Gbest + 0.05) / (Group[i].fitness(j) + 0.05))
            flag_index = int(np.floor(2 * np.random.rand() + 1))
            var_flag = vec_flag(flag_index)
            for k in np.arange(1, var_nvars + 1).reshape(-1):
                Group[i].Position[j, k] = Group[i].Position(j, k) + var_flag * np.random.rand() * gama * (
                            best_pos[i](k) - Group[i].Position(j, k)) + np.random.rand() * alpha * var_flag * (
                                                      S(i) * vec_Xbest(k) - Group[i].Position(j, k))

    return Group


def fun_checkpoisions(dim, Group, var_n_gases, var_n_types, var_down, var_up):
    Lb = var_down * np.ones((1, dim))
    # Upper bounds
    Ub = var_up * np.ones((1, dim))
    for j in np.arange(1, var_n_types + 1).reshape(-1):
        for i in np.arange(1, var_n_gases / var_n_types + 1).reshape(-1):
            isBelow1 = Group[j].Position[i,:]< Lb
            isAboveMax = (Group[j].Position[i,:] > Ub)
            if isBelow1 == True:
                Group[j].Position[i, :] = Lb
            else:
                if np.find(isAboveMax == True):
                    Group[j].Position[i, :] = Ub

    return Group


def Evaluate(objfunc, var_n_types, var_n_gases, X, Xnew, init_flag):
    if init_flag == 1:
        for j in np.arange(1, var_n_gases / var_n_types + 1).reshape(-1):
            X.fitness[j] = objfunc(X.Position[j,:])
    else:
        for j in np.arange(1, var_n_gases / var_n_types + 1).reshape(-1):
            temp_fit = objfunc(Xnew.Position[j,:])
            if temp_fit < X.fitness(j):
                X.fitness[j] = temp_fit
                X.Position[j, :] = Xnew.Position[j,:]

            best_fit, index_best = np.amin(X.fitness)
            best_pos = X.Position[index_best,:]
            return X, best_fit, best_pos


def update_variables(var_iter, var_niter, K, P, C, var_n_types, var_n_gases):
    T = np.exp(- var_iter / var_niter)
    T0 = 298.15
    i = 1
    N = var_n_gases / var_n_types
    S = []
    for j in np.arange(1, var_n_types + 1).reshape(-1):
        K[j] = K(j) * np.exp(- C(j) * (1 / T - 1 / T0))
        S[np.arange[i, i + N + 1], :] = P[np.arange(i, i + N + 1),:] *K[j]
        i = j * N + 1
        if i + N > var_n_gases:
            i = j * N
    return S


def worst_agents(X, M1, M2, dim, G_max, G_min, var_n_gases, var_n_types):
    # Rank and select number of worst agents eq.(11)
    X_sort, X_index = np.sorted(X.fitness, 'descend')
    M1N = M1 * var_n_gases / var_n_types
    M2N = M2 * var_n_gases / var_n_types
    Nw = np.round(np.multiply((M2N - M1N), np.random.rand(1, 1)) + M1N)
    for k in np.arange(1, Nw + 1).reshape(-1):
        X.Position[X_index[k], :] = G_min + np.random.rand(1, dim) * (G_max - G_min)

    return X



def HGSO(X, objfunc, var_down, var_up, var_niter):
    var_n_gases = 35
    var_n_types = 5

    dim = X.shape[1]
    # constants in eq (7)
    l1 = 0.005
    l2 = 100
    l3 = 0.01
    # constants in eq (10)
    alpha = 1
    beta = 1
    # constant in eq (11)
    M1 = 0.1
    M2 = 0.2
    for t in range(var_niter):
        # paramters setting in eq. (7)
        K = l1 * np.random.rand(var_n_types, 1)
        P = l2 * np.random.rand(var_n_gases, 1)
        C = l3 * np.random.rand(var_n_types, 1)

        # The population agents are divided into equal clusters with the same Henry痴 constant value
        Group = Create_Groups(var_n_gases, var_n_types, X)
        best_fit =[]
        best_pos =[]
        ct = time.time()
        # Compute cost of each agent
        for i in np.arange(1, var_n_types + 1).reshape(-1):
            Group[i], best_fit[i], best_pos[i] =Evaluate(objfunc,var_n_types, var_n_gases, Group[i], 0, 1)

        var_Gbest, var_gbest = np.amin(best_fit)
        vec_Xbest = best_pos[var_gbest]
        vec_Gbest_iter=[]
        for var_iter in np.arange(1, var_niter + 1).reshape(-1):
            S = update_variables(var_iter, var_niter, K, P, C, var_n_types, var_n_gases)
            Groupnew = update_positions(Group, best_pos, vec_Xbest, S, var_n_gases, var_n_types, var_Gbest, alpha, beta,
                                        dim)
            Groupnew = fun_checkpoisions(dim, Groupnew, var_n_gases, var_n_types, var_down, var_up)
            for i in np.arange(1, var_n_types + 1).reshape(-1):
                Group[i], best_fit[i], best_pos[i] =Evaluate(objfunc,var_n_types, var_n_gases, Group[i], Groupnew[i], 0)
                Group[i] = worst_agents(Group[i], M1, M2, dim, var_up, var_down, var_n_gases, var_n_types)
            var_Ybest, var_index = np.amin(best_fit)
            vec_Gbest_iter[var_iter] = var_Ybest
            if var_Ybest < var_Gbest:
                var_Gbest = var_Ybest
                vec_Xbest = best_pos[var_index]
        ct = time.time() -ct

        return vec_Xbest, var_Gbest, vec_Gbest_iter,  ct
