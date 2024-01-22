import numpy as np
import matplotlib.pyplot as plt

class FECHamming():
    def __init__(self, n_parity, soft_decision= True):

        self.n_parity = n_parity
        self.n, self.k, self.G,self.H,self.P,self.R = self.hamm_params()

        if soft_decision:

            self.valid_codewords = np.zeros((2**self.k, self.n))

            for i in range(0,2**self.k):
                bits = np.array([int(j) for j in np.binary_repr(i, self.k)])
                self.valid_codewords[i] = self.hamm_encode(bits)
            



    def hamm_params(self):

        if (self.n_parity<3):
            raise ValueError('number of parity bits must be >2')
        
        n = 2**self.n_parity-1
        k = n - self.n_parity


        G = np.zeros((k,n), dtype=int)
        H = np.zeros((self.n_parity,n), dtype=int)
        P = np.zeros((self.n_parity,k), dtype=int)
        R = np.zeros((k,n), dtype=int)

        for i in range(1,n+1):
            bits = [int(j) for j in np.binary_repr(i, self.n_parity)]
            H[:,i-1] = np.array(bits)

        H1 = np.zeros((1,self.n_parity),dtype=int)
        H2 = np.zeros((1,self.n_parity),dtype=int)

        for i in range(self.n_parity):
            id1 = 2**i-1
            id2 = n-i-1
            H[:, [id1, id2]] = H[:, [id2, id1]]
        
        P = H[:,:k]
        G[:,:k] = np.diag(np.ones(k))
        G[:,k:] = P.T
        R[:,:k] = np.diag(np.ones(k))

        return n,k,G,H,P,R
    
    def hamm_encode(self, packet_data_bits):
        
        if len(packet_data_bits)<self.k:
            raise ValueError('Error: bits must be of length k')
        
        if(np.dtype(packet_data_bits[0]) != int):
            raise ValueError('Error: Invalid data type. Input must be a vector of ints')
        
        codeword = np.matmul(packet_data_bits, self.G)%2 

        return codeword
    

    
    
    def hamm_decode(self, received_data_bits):
        
        #----------Hard decision-------------------
        if len(received_data_bits)<self.n:
            raise ValueError(f'Error: bits must be of length {self.n}')
        
        if(np.dtype(received_data_bits[0]) != int):
            raise ValueError('Error: Invalid data type. Input must be a vector of ints')
        
        syndrome = np.matmul(self.H, received_data_bits.T)%2

        error_location = int(''.join([str(elem) for elem in syndrome]),2)

        H1 = self.H[:,error_location-1]

        decoded_error_pos = int(''.join([str(elem) for elem in H1]),2)

        if error_location:
            received_data_bits[decoded_error_pos-1] = (received_data_bits[decoded_error_pos-1]+1)%2

        decoded_bits = np.matmul(self.R, received_data_bits.T).T %2

        return decoded_bits
    

    def hamm_soft_decode(self, received_symbols):

        if len(received_symbols)!=self.n:
            raise ValueError(f'Error: bits must be of length {self.n}')
        
        euclidean_distance = np.zeros((2**self.k,))
        for i in range(2**self.k):
            euclidean_distance[i] = np.linalg.norm(received_symbols-self.valid_codewords[i])

        min_id = np.argmin(euclidean_distance)
        detected_code = self.valid_codewords[min_id]

        decoded_bits = self.hamm_decode(detected_code.astype(int))

        return decoded_bits

        
        

    




    
    

            