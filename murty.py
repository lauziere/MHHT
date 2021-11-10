
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

def getkBestNoRankHung(matNR, k_bestNR):
        
    n = matNR.shape[0]
    matNRcols = np.arange(n)
    
    nextMat = matNR.copy()
    nextMatcols = matNRcols.copy()

    origMatrix = matNR.copy()
    origMatrixcols = matNRcols.copy()
    
    proxy_InfNR = 1e6
    
    i = 0
    
    all_solutions = []
    all_colnames = []
    all_objectives = []
    
    fullMats = []
    fullcols = []
    fullObjs = []
    
    partialSols = []
    partitionsAll = []
    partitionsAllcols = []
    
    colsToAddAll = []
    colsToAdd = []
    
    n_possible = np.math.factorial(n)
    
    solution, objval = parseClueOutput(matNR)
    
    all_solutions.append(solution)
    all_objectives.append(objval)
    
    curr_solution = solution
    full_solution = curr_solution
    
    while i < k_bestNR-1:

        idx = np.vstack([np.arange(curr_solution.shape[0]), np.argmax(curr_solution>0,axis=1)]).T
        idx = np.delete(idx, idx.shape[0]-1, axis=0)

        if idx.shape[0] != 0:
            
            idxStrike = [idx[:z+1, :] for z in range(idx.shape[0]-1)]
            idxmaxSubs = [idx[z, :] for z in range(idx.shape[0])]
            
        else:
            
            idxStrike = np.inf
            idxmaxSubs = idx

        matSub = PartitionAndInsertInf(idx, nextMat, idxmaxSubs)
        matSubcols = [nextMatcols for z in range(idx.shape[0])]

        matSub, matSubcols = strikeRwsCols(matSub, matSubcols, idxStrike)

        matCheck1 = [z for z in range(len(matSub)) if np.any(((matSub[z]==1e6).sum(axis=1))==matSub[z].shape[1])]
        matCheck2 = [z for z in range(len(matSub)) if np.any(((matSub[z]==1e6).sum(axis=0))==matSub[z].shape[0])]
        matCheck = matCheck1 + matCheck2

        if len(matCheck) > 0:
            
            matSub = [matSub[z] for z in range(len(matSub)) if z not in matCheck]
            matSubcols = [matSubcols[z] for z in range(len(matSubcols)) if z not in matCheck]

        if len(matSub) > 0:
            
            partitionsAll = partitionsAll + matSub
            partitionsAllcols = partitionsAllcols + matSubcols

            algoList = [parseClueOutputInf(matSub[z]) for z in range(len(matSub))]

            partialSols = partialSols + [algoList[z][0] for z in range(len(algoList))]
            colsToAddAll = colsToAddAll + [list(set(np.arange(n)) - set(matSubcols[z])) for z in range(len(matSub))]

            reconstructedPartition = reconstructPartition(algoList, idx, idxStrike, curr_solution, nextMat)

            if reconstructedPartition[0].shape[0] != origMatrix.shape[0]:

                reconstructedPartition = reconstructInitial(reconstructedPartition, colsToAdd, full_solution)

            fullObjsTmp = [origMatrix[np.argwhere(reconstructedPartition[z]==1)[:,0],np.argwhere(reconstructedPartition[z]==1)[:,1]].sum() for z in range(len(reconstructedPartition))]

            fullMats.extend(reconstructedPartition)
            fullObjs.extend(fullObjsTmp)


        idxOpt = np.argmin(fullObjs)

        i += 1

        full_solution = fullMats[idxOpt]
        curr_solution = partialSols[idxOpt]
        nextMat = partitionsAll[idxOpt]
        nextMatcols = partitionsAllcols[idxOpt]
        colsToAdd = colsToAddAll[idxOpt]

        all_solutions.append(fullMats[idxOpt])
        all_colnames.append(partitionsAllcols[idxOpt])

        all_objectives.append(fullObjs[idxOpt])

        del fullMats[idxOpt]
        del fullObjs[idxOpt]
        del partialSols[idxOpt]
        del partitionsAll[idxOpt]
        del partitionsAllcols[idxOpt]
        del colsToAddAll[idxOpt]

        if (len(all_solutions) == n_possible) and (k_bestNR > n_possible):

            break

    return np.array(all_solutions), np.array(all_objectives)

def PartitionAndInsertInf(idx, nextMat, idxmaxSubs):
    
    if idx.shape[0] != 0:
        
        matSub = [nextMat.copy() for i in range(idx.shape[0])]

        for x in range(len(matSub)):

            tmpMat = matSub[x]
            tmpidxMax = idxmaxSubs[x]
            tmpMat[tmpidxMax[0], tmpidxMax[1]] = 1e6

            matSub[x] = tmpMat
        
    else:
        
        matSub = nextMat
        tmpidxMax = idxmaxSubs
        
        matSub[tmpidxMax[0],tmpidxMax[1]] = 1e6
    
    return matSub

def strikeRwsCols(matSub, matSubcolnames, idxStrike):

    if type(matSub) == list:

        for x in range(len(matSub)):

            if x >= 2:

                tmpMat = matSub[x]
                tmpMatcolnames = matSubcolnames[x]

                rowsToRem = idxStrike[x - 1][:, 0]
                colsToRem = idxStrike[x - 1][:, 1]

                tmpMat = np.delete(tmpMat, rowsToRem, axis=0)
                tmpMat = np.delete(tmpMat, colsToRem, axis=1)
                tmpMatcolnames = np.delete(tmpMatcolnames, colsToRem, axis=0)

                matSub[x] = tmpMat
                matSubcolnames[x] = tmpMatcolnames

            elif x == 1:

                tmpMat = matSub[x]
                tmpMatcolnames = matSubcolnames[x]

                rowsToRem = idxStrike[x - 1][0, 0]
                colsToRem = idxStrike[x - 1][0, 1]

                tmpMat = np.delete(tmpMat, rowsToRem, axis=0)
                tmpMat = np.delete(tmpMat, colsToRem, axis=1)
                tmpMatcolnames = np.delete(tmpMatcolnames, colsToRem, axis=0)

                matSub[x] = tmpMat
                matSubcolnames[x] = tmpMatcolnames

            else:

                pass
        
    else:

        matSub = [matSub]
        matSubcolnames = [matSubcolnames]

    return matSub, matSubcolnames

def parseClueOutput(mat):
    
    rows, cols = linear_sum_assignment(mat)
    n = rows.shape[0]
    
    tmp = np.zeros((n,n),'int')
    tmp[rows,cols]=1
    cost = mat[rows,cols].sum()
    
    return [tmp, cost]

def parseClueOutputInf(mat):

    rows, cols = linear_sum_assignment(mat)
    n = rows.shape[0]
    
    tmp = np.zeros((n,n),'int')
    tmp[rows,cols]=1
    cost = mat[rows,cols].sum()
    
    return [tmp, cost]

def reconstructPartition(algoList, idx, idxStrike, curr_solution, nextMat):

    matMemory = [curr_solution[:z+1] for z in range(idx.shape[0]-1)]

    reconstructedPartition = []

    for x in range(len(algoList)):

        assignmSol = algoList[x][0]

        if assignmSol.shape[0] == nextMat.shape[0]:

            assignmFull = assignmSol

            reconstructedPartition.append(assignmFull)

        else:

            if assignmSol.shape[0] == nextMat.shape[0] - 1:

                colsToAddPartition = idxStrike[nextMat.shape[1] - assignmSol.shape[1]-1][:,1]

            else:

                colsToAddPartition = idxStrike[nextMat.shape[1] - assignmSol.shape[1]-1][:,1]

            emptyMat = np.zeros((assignmSol.shape[0], len(colsToAddPartition)), 'int')
            assignmFull = np.concatenate([assignmSol, emptyMat],axis=1)

            colOrder = np.argsort(np.concatenate([np.arange(assignmSol.shape[1]), np.sort(colsToAddPartition)-np.arange(colsToAddPartition.shape[0])-1]), kind='mergesort')

            assignmFull = assignmFull[:, colOrder]

            assignmFull = np.concatenate([matMemory[nextMat.shape[1] - assignmSol.shape[1]-1], assignmFull], axis=0)

            reconstructedPartition.append(assignmFull)

    return reconstructedPartition

def reconstructInitial(reconstructedPartition, colsToAdd, full_solution):

    reconstructedInitial = []

    if len(colsToAdd) > 1:

        matMemory = full_solution[:len(colsToAdd), :]

        for x in range(len(reconstructedPartition)):

            assignmSol = reconstructedPartition[x]

            emptyMat = np.zeros((assignmSol.shape[0], len(colsToAdd)), 'int')

            assignmFull = np.concatenate([assignmSol, emptyMat],axis=1)
            colOrder = np.argsort(np.concatenate([np.arange(1,assignmSol.shape[1]+1), np.sort(colsToAdd)-np.arange(len(colsToAdd))]), kind='mergesort')

            assignmFull = assignmFull[:, colOrder]

            assignmFull = np.concatenate([matMemory, assignmFull], axis=0)

            reconstructedInitial.append(assignmFull)

    else:

        matMemory = full_solution[0,:]

        reconstructedInitial = []

        for x in range(len(reconstructedPartition)):

            assignmSol = reconstructedPartition[x]

            emptyMat = np.zeros((assignmSol.shape[0], len(colsToAdd)), 'int')

            assignmFull = np.concatenate([assignmSol, emptyMat],axis=1)

            col = np.concatenate([np.arange(1, assignmSol.shape[1]+1), np.sort(colsToAdd)-np.arange(len(colsToAdd))])
            colOrder = np.argsort(col, kind='mergesort')
            assignmFull = assignmFull[:,colOrder]
            assignmFull = np.vstack([matMemory, assignmFull])

            reconstructedInitial.append(assignmFull)

    return reconstructedInitial
 
def Murty_mat(in_arr, out_arr, b, d):
    
    # Construct C. 
    n1 = in_arr.shape[0]
    n2 = out_arr.shape[0]
    
    CC = np.empty((n1+n2, n1+n2))
    
    # Block A:
    A = distance.cdist(in_arr, out_arr)

    CC[:n1, :n2] = A
    
    # Block B:
    B = 1e6*np.ones((n1, n1))
    np.fill_diagonal(B, d)
    
    CC[:n1, n2:] = B
    
    # Block C:
    C = 1e6*np.ones((n2, n2))
    np.fill_diagonal(C, b)
    
    CC[n1:, :n2] = C
    
    # Block D
    D = A.T
    # D = 1e6*np.ones((n1,n2)).T
    
    CC[n1:, n2:] = D
    
    return CC

def Murty(C, k):
    
    solutions, k_best_costs = getkBestNoRankHung(C,k)
    
    rows = [np.arange(C.shape[0]) for i in range(k)]
    cols = solutions.argmax(axis=2)

    return k_best_costs, rows, cols

def get_unique_hyps(col_sets, n):
    
    col_sets = col_sets[:,:n]
        
    checked = [col_sets[0]]

    for k in range(col_sets.shape[0]):
        
        this_pred = col_sets[k]
        matches = 1*np.array([np.array_equiv(this_pred, checked[z]) for z in range(len(checked))])

        if matches.sum() == 0:
            checked.append(this_pred)

    return np.array(checked)