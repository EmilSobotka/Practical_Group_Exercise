import numpy as np

def viterbi(obs, states, pi, A, b):
    N = len(states)  
    T = len(obs)  
    
    #Initialize a path probability matrix
    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)
    
    #Initialization step
    for q in range(N):
        viterbi[q, 0] = pi[q] * b[q][obs[0]]

    #Recursion step
    for t in range(1, T):  
        for q in range(N):  
            maxProb, bestState = max(
                (viterbi[q_prime, t - 1] * A[q_prime][q] * b[q][obs[t]], q_prime)
                for q_prime in range(N)
            )
            viterbi[q, t] = maxProb
            backpointer[q, t] = bestState
    
    bestPathProb = max(viterbi[q, T - 1] for q in range(N))
    bestLastState = np.argmax(viterbi[:, T - 1])

    bestPath = [bestLastState]
    for t in range(T - 1, 0, -1):
        bestPath.append(backpointer[bestPath[-1], t])
    bestPath.reverse()

    return bestPathProb, bestPath
