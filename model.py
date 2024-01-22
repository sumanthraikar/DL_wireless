import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sys
sys.path.insert(0, '/home/sumanthraikar/Desktop/Casual Learning/pyadi_iio')
import adi 
from utils import Modulation, Demodulator


class CustomSigmoid(nn.Module):

    def __init__(self,alpha=2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.alpha*x))

class Autoencoder(nn.Module):

    def __init__(self,act_fn, no_of_inputs, no_of_outputs,power=1):
        super(Autoencoder,self).__init__()
        self.act_fn = act_fn
        self.k = no_of_inputs
        self.n = no_of_outputs
        self.code_rate = np.log2(no_of_inputs)/no_of_outputs
        self.P = power
        

        self.encoder = nn.Sequential(
            nn.Linear(self.k, self.n),
            self.act_fn(),
            nn.Linear(self.n,self.n)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.n,self.k),
            self.act_fn(),
            #nn.Linear(self.M,self.M),
            #nn.ReLU(),
            nn.Linear(self.k,self.k),
            
        )

    def encoded_symbol(self,X):
        return self.encoder(X)

    def EbNo_to_noise(self,ebno):
        rate = self.code_rate
        snr = 2*rate*ebno
        noise_var = 1/snr

        return noise_var

    def decoded_symbol(self,X):
        return self.decoder(X)

    def forward(self,X,ebno):
        X = self.encoder(X)
        #print(torch.norm(X,2,1).size())
        X = ((self.n*self.P)**0.5*X)/torch.norm(X,2,1)[:,None]
        noise_var = self.EbNo_to_noise(ebno)
        noise = Variable(torch.randn(X.size())*(noise_var**0.5))
        X+=noise
        X = self.decoder(X)

        return X 

class NN_Symbol_Mapper(nn.Module):

    def __init__(self, act_fn, n, k, threshold=0.5):
        super(NN_Symbol_Mapper, self).__init__()
        self.act_fn = act_fn
        self.n = n
        self.k  = k
        self.threshold = threshold

        self.mapper = nn.Sequential(
            nn.Linear(2**self.k, self.n), 
            self.act_fn(),
            nn.Linear(self.n, self.n), 
            nn.Sigmoid()

        )

    def forward(self,X):
        X = self.mapper(X)
        # X = torch.where(X>self.threshold, 1.0, 0.0)
        return X
    

class Simple_nn_enc(nn.Module):

    def __init__(self, act_fn, n, k, threshold=0.5):
        super(Simple_nn_enc, self).__init__()
        self.act_fn = act_fn
        self.n = n
        self.k  = k
        self.threshold = threshold

        self.mapper = nn.Sequential(
            nn.Linear(2**self.k, self.n), 
            self.act_fn(),
            nn.Linear(self.n, self.n), 
            # nn.Sigmoid()
            CustomSigmoid(alpha=1)
            

        )

    def forward(self,X):
        X = self.mapper(X)
        return X
    
class Simple_nn_dec(nn.Module):

    def __init__(self, act_fn,n,k):
        super(Simple_nn_dec,self).__init__()
        self.act_fn = act_fn
        self.n = n
        self.k = k

        self.s2b_mapper = nn.Sequential(
            nn.Linear(self.n, 2**self.k),
            self.act_fn(),
            nn.Linear(2**self.k, 2**self.k), 
            # nn.LogSoftmax()
        )

    def forward(self,X):
        
        X = X/torch.norm(X,2,1)[:,None]
        X = self.s2b_mapper(X)
        

        return X



class NN_Symbol_demapper(nn.Module):

    def __init__(self, act_fn,n,k):
        super(NN_Symbol_demapper,self).__init__()
        self.act_fn = act_fn
        self.n = n
        self.k = k

        self.s2b_mapper = nn.Sequential(
            nn.Linear(self.n, 2**self.k),
            self.act_fn(),
            nn.Linear(2**self.k, 2**self.k)
            # nn.LogSoftmax()
        )

    def forward(self,X):
        
        X = X/torch.norm(X,2,1)[:,None]
        X = self.s2b_mapper(X)
        

        return X
    



class ED_learner(nn.Module):

    def __init__(self,act_fn, n, k, threshold):
        super(ED_learner,self).__init__()
        self.act_fn = act_fn
        self.n = n
        self.k = k
        self.threshold = Variable(torch.Tensor([threshold])) 
        self.n_act = int(np.log2(n))

        self.encoder = nn.Sequential(
                nn.Linear(self.k,self.n_act),
                self.act_fn(),
                #nn.Linear(self.M,self.M),
                #nn.ReLU(),
                nn.Linear(self.n_act,self.n_act),
                
            )

        self.decoder = nn.Sequential(
                nn.Linear(self.n,self.k),
                self.act_fn(),
                #nn.Linear(self.M,self.M),
                #nn.ReLU(),
                nn.Linear(self.k,self.k),
                nn.Softmax()
            )
        
    def pluto_tx_rx(self,bits, sps=1):
        sample_rate = 1e6
        center_freq = 915e6 
        

        modulator = Modulation(sample_rate=sample_rate,center_freq=center_freq)
        demodulator = Demodulator(sample_rate=sample_rate,center_freq=center_freq,sps=sps,data_length=len(bits))
        modulator.OOK(bits,sps=sps,repeat_tx=True,plot=False)

        rx_bits=demodulator.OOK_demodulate(plot=False)
        modulator.sdr.tx_destroy_buffer()

        if len(rx_bits)>0 and (rx_bits!=None).any():
            return torch.from_numpy(rx_bits.astype('float32'))
        else:
            print('Outliers detected')
            return torch.zeros_like(rx_bits)
            
        
    def one_hot_converter(self,array):
        batch_size, code_length = array.shape
        one_hot = np.zeros((batch_size, 2**code_length))
        if (array!=None).any():
            b = [''.join([str(int(j)) for j in i]) for i in array]
            c = [int(i,2) for i in b]
            one_hot[np.arange(batch_size), c] = 1

            return torch.from_numpy(one_hot.astype('float32'))

        else:
            return torch.from_numpy(one_hot.astype('float32'))
        

    def forward(self,X, sps):
        X = self.encoder(X)
        X = torch.where(X > self.threshold, 1.0, 0.0)

        with torch.no_grad():
            X = X.reshape((-1,))
            b = np.copy(X.numpy())
            
            rx_bits = self.pluto_tx_rx(bits=b, sps=sps)
            rx_bits = np.reshape(rx_bits,(1,-1))
            
            # print(f'rx_data{rx_bits.shape}')
            X = self.one_hot_converter(rx_bits.numpy())
        
        X = self.decoder(X)
        return X
        
        


class alternate_ED_learner(nn.Module):

    def __init__(self,act_fn, n, k, threshold):
        super(alternate_ED_learner,self).__init__()
        self.act_fn = act_fn
        self.n = n
        self.k = k
        self.threshold = Variable(torch.Tensor([threshold])) 
        self.n_act = int(np.log2(n))

        self.encoder = nn.Sequential(
                nn.Linear(self.k,self.n_act),
                self.act_fn(),
                # nn.Linear(self.n_act,self.n_act),
                # self.act_fn(),
                nn.Linear(self.n_act,self.n_act),
                nn.Sigmoid()
                
            )

        self.decoder = nn.Sequential(
                nn.Linear(self.n_act,self.k),
                self.act_fn(),
                # nn.Linear(self.k,self.k),
                # self.act_fn(),
                nn.Linear(self.k,self.k),
                # nn.Softmax()
            )
        
    def pluto_tx_rx(self,bits, sps=1):
        sample_rate = 1e6
        center_freq = 915e6 
        

        modulator = Modulation(sample_rate=sample_rate,center_freq=center_freq)
        demodulator = Demodulator(sample_rate=sample_rate,center_freq=center_freq,sps=sps,data_length=len(bits))
        modulator.OOK(bits,sps=sps,repeat_tx=True,plot=False)

        rx_bits=demodulator.OOK_demodulate(plot=False)
        modulator.sdr.tx_destroy_buffer()

        if len(rx_bits)>0 and (rx_bits!=None).any():
            return torch.from_numpy(rx_bits.astype('float32'))
        else:
            print('Outliers detected')
            return torch.zeros_like(rx_bits)
            
        
    def one_hot_converter(self,array):
        batch_size, code_length = array.shape
        one_hot = np.zeros((batch_size, 2**code_length))
        if (array!=None).any():
            b = [''.join([str(int(j)) for j in i]) for i in array]
            c = [int(i,2) for i in b]
            one_hot[np.arange(batch_size), c] = 1

            return torch.from_numpy(one_hot.astype('float32'))

        else:
            return torch.from_numpy(one_hot.astype('float32'))
        
    def dec_train(self,X,sps):
        
        #----------Switch ON encoder grads--------------
        for param in self.encoder.parameters():
            param.requires_grad = True

        #----------Switch ON decoder grads--------------
        for param in self.decoder.parameters():
            param.requires_grad = True

        X = self.encoder(X)
        X = torch.where(X > self.threshold, 1.0, 0.0)


        #----------Actual transmission happens here-------
        with torch.no_grad():
            X = X.reshape((-1,))
            b = np.copy(X.numpy())
            
            rx_bits = self.pluto_tx_rx(bits=b, sps=sps)
            rx_bits = np.reshape(rx_bits,(1,-1))
        #------------------------------------------------
        
        X = self.one_hot_converter(rx_bits.numpy())
        X = self.decoder(X)
        
        return X
        
        
    
    def enc_train(self, X,sps):
        #----------Switch off decoder grads------------- 
        for param in self.decoder.parameters():
            param.requires_grad = True

        #----------Switch ON encoder grads--------------
        for param in self.encoder.parameters():
            param.requires_grad = True

        X = self.encoder(X)
        X = torch.where(X > self.threshold, 1.0, 0.0)


        #----------Actual transmission happens here-------
        with torch.no_grad():
            X = X.reshape((-1,))
            b = np.copy(X.numpy())
            
            rx_bits = self.pluto_tx_rx(bits=b, sps=sps)
            rx_bits = np.reshape(rx_bits,(1,-1))
        #------------------------------------------------
        
        X = self.one_hot_converter(rx_bits.numpy()) #remove this
        X = self.decoder(X)

        return X
    
    def forward(self,X,train_dec, sps):
        if train_dec:
            X = self.dec_train(X,sps)
        else:
            X = self.enc_train(X,sps)

        return X

            

    
    def decoded_symbol(self,X):
    
        with torch.no_grad():
            X = self.decoder(X)
        return X
    
    def encoded_symbol(self,X):

        with torch.no_grad():
            X = torch.where(self.encoder(X)>self.threshold, 1.0,0.0)
        
        return X
        

class alternate_ED_learner_more_batchsize(nn.Module):

    def __init__(self,act_fn, n, k, threshold):
        super(alternate_ED_learner_more_batchsize,self).__init__()
        self.act_fn = act_fn
        self.n = n
        self.k = k
        self.threshold = Variable(torch.Tensor([threshold])) 
        self.n_act = int(np.log2(n))
        self.k_act = int(np.log2(k))

        #----------------Dataset generation--------------------------
        bits_dataset = np.zeros((self.k, self.k_act))
        for i in range(self.k):
            bits = np.array([int(j) for j in np.binary_repr(i, self.k_act)])
            bits = np.reshape(bits,(1,-1))
            bits_dataset[i] = bits
        self.X = self.one_hot_converter(bits_dataset)
        self.Y = self.one_hot_converter(bits_dataset)

        


        self.encoder = nn.Sequential(
                nn.Linear(self.k,self.n_act),
                self.act_fn(),
                #nn.Linear(self.M,self.M),
                #nn.ReLU(),
                nn.Linear(self.n_act,self.n_act),
                
            )

        self.decoder = nn.Sequential(
                nn.Linear(self.n,self.k),
                self.act_fn(),
                #nn.Linear(self.M,self.M),
                #nn.ReLU(),
                nn.Linear(self.k,1),
                # nn.Softmax()
            )
        
    def pluto_tx_rx(self,bits, sps=1):
        sample_rate = 1e6
        center_freq = 915e6 
        

        modulator = Modulation(sample_rate=sample_rate,center_freq=center_freq)
        demodulator = Demodulator(sample_rate=sample_rate,center_freq=center_freq,sps=sps,data_length=len(bits))
        modulator.OOK(bits,sps=sps,repeat_tx=True,plot=False)

        rx_bits=demodulator.OOK_demodulate(plot=False)
        modulator.sdr.tx_destroy_buffer()

        if len(rx_bits)>0 and (rx_bits!=None).any():
            return torch.from_numpy(rx_bits.astype('float32'))
        else:
            print('Outliers detected')
            return torch.zeros_like(rx_bits)
            
        
    def one_hot_converter(self,array):
        batch_size, code_length = array.shape
        one_hot = np.zeros((batch_size, 2**code_length))
        if (array!=None).any():
            b = [''.join([str(int(j)) for j in i]) for i in array]
            c = [int(i,2) for i in b]
            one_hot[np.arange(batch_size), c] = 1

            return torch.from_numpy(one_hot.astype('float32'))

        else:
            return torch.from_numpy(one_hot.astype('float32'))
        
    
    def enc_dec_train(self,sps):
        
        #-------------------Encoding------------------------
        X_ = self.encoder(self.X)
        X_ = torch.where(X_ > self.threshold, 1.0, 0.0)


        #----------Actual serial transmission happens here-------
        with torch.no_grad():
            received_dataset = np.zeros((self.k, self.n_act))
            for g,serial_data in zip(range(self.k),X_):
                b = np.copy(serial_data.numpy())
                rx_bits = self.pluto_tx_rx(bits=b, sps=sps)
                received_dataset[g] = rx_bits
            
        #----------------decoding--------------------------------
        
        X_ = self.one_hot_converter(received_dataset)
        X_ = self.decoder(X_)

        return X_
    
    def forward(self,sps):
        
        X = self.enc_dec_train(sps=sps)

        return X

            

    
    def decoded_symbol(self,X):
    
        with torch.no_grad():
            X = self.decoder(X)
        return X
    
    def encoded_symbol(self,X):

        with torch.no_grad():
            X = torch.where(self.encoder(X)>self.threshold, 1.0,0.0)
        
        return X



class Shallow_NN(nn.Module):

    def __init__(self,act_fn, n, k):
        super(Shallow_NN,self).__init__()
        self.act_fn = act_fn
        self.n = n
        self.k = k
        

        self.decoder = nn.Sequential(
                nn.Linear(self.n,self.k),
                self.act_fn(),
                #nn.Linear(self.M,self.M),
                #nn.ReLU(),
                nn.Linear(self.k,self.k),
                nn.Softmax()
            )
        
    def forward(self, X):
        X = self.decoder(X)
        return X
    
    def decoded_symbol(self,X):
        return self.decoder(X)
        

