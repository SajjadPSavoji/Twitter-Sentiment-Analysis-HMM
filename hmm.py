import numpy as np
class HMM():
    '''
    A is the transition matrix where A[i][j] is the probability of transition from state i to state j
    B is the observation matrix where B[j][o] is the probability of observation of "o" in the state j
    P is the initial probability array whare P[j] is the initial prob of being in the state j
    N is the number of hidden states in HMM

    !!! A and B should be numpy 2d-arrays
    '''
    def __init__(self , A , B , P):
        self.A = A
        self.B = B
        self.P = P
        self.N = A.shape[0]

    def forward(self , O):
        '''
        O should contain indexes of emissions
        !!! O should be numpy 1d-arrayd

        this function returns the probability of observation sequence "O" using forward algorithm
        p(O) = a_t(1) + a_t(2) + ...+ a_t(j) + ... + a_t(N) where : 
            a_t(j) = p(O , s_t = j | A , B)
            s_t is the hidden state in time step "t"
            N is the number of hidden states

        the reqursion formula is as followed : 
            a_{t}(j) = sum over i { a_{t-1}(i) * a[i][j] * b[j][o_t] }
        '''
        T = O.shape[0]
        a = np.zeros((T + 1 , self.N))

        # initial first timestep -----> t = 1

        for j in range(self.N):
            a[1][j] = self.P[j] * self.B[j][O[0]]

        for t in range(2 , T + 1) : 
            for j in range(self.N) : 
                for i in range(self.N) : 
                    a[t][j] += a[t-1][i] * self.A[i][j] * self.B[j][O[t-1]]
        
        return np.sum(a[T][:])

if __name__ == "__main__":


    # problem 5 ---------------------------------------------------------------------------

    # A = np.array([[0.7 , 0.5], [0.3 , 0.6]])
    # B = np.array([[0.1 , 0.4 , 0.5] , [0.6 , 0.3 , 0.1]])
    # P = np.array([0.6 , 0.4])

    # O1 = np.array([1 , 2 , 0 , 2 , 0 , 2 , 1 , 2 , 1 , 0 , 1 , 2 , 1 , 0 ])
    # O2 = np.array([0 , 2 , 0 , 1 , 0 , 2 , 1 , 0 , 1 , 0 , 2 , 0 , 1 , 2 ])
    # O3 = np.array([0])

    # hmm = HMM(A , B , P)
    # print(hmm.forward(O1))
    # print(hmm.forward(O2))
    # print(hmm.forward(O3))

    # ------------------------------------------------------------------------------------

    # problem 6 --------------------------------------------------------------------------
    A1 = np.array([[0.6 , 0.4], [0 , 1]])
    B1 = np.array([[0.45 , 0.55] , [0.5 , 0.5]])

    A2 = np.array([[0.15 , 0.85], [0 , 1]])
    B2 = np.array([[0.4 , 0.6] , [0.7 , 0.3]])

    P = np.array([0.5 , 0.5])

    O = np.array([0 , 1 , 0])
    
    hmm1 = HMM(A1 , B1 , P)
    hmm2 = HMM(A2 , B2 , P)

    print(hmm1.forward(O))
    print(hmm2.forward(O))

