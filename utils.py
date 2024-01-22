import numpy as np
import sys
sys.path.insert(0, '/home/sumanthraikar/Desktop/Casual Learning/pyadi_iio')
import adi 
import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
import torch
from torch.utils.data import Dataset

class one_hot_dataset(Dataset):

    def __init__(self,k,size,onehot=True,power_required = False,power_range=(1,3)):
        self.power_required = power_required
        self.M = 2**k
        if onehot:

            if power_required:

                low,high = power_range
                xy = torch.eye(self.M,dtype=torch.float32)[np.random.choice(self.M,size)]
                power_levels = torch.arange(low,high,dtype = torch.float64)
                p = torch.ones_like(power_levels)/len(power_levels)
                idx = p.multinomial(list(xy.size())[0],replacement=True)
                b = power_levels[idx]
                b = torch.reshape(b,(list(xy.size())[0],1))
                xy = torch.hstack((xy,b))
                self.x = xy[:,:-1]
                self.y = xy[:,:-1]
                self.power = xy[:,-1]
            
            else:
                xy = torch.eye(self.M,dtype=torch.float32)[np.random.choice(self.M,size)]
                self.x = xy
                self.y = xy
            

        else:
            xy= torch.randint(10,(size,1))
            self.x = xy
            self.y = xy
        self.n_samples = xy.shape[0]



    def __getitem__(self,index):
        #dataset[index]
        if self.power_required:
            return self.x[index], self.y[index], self.power[index]

        else:
            return self.x[index], self.y[index]

    def __len__(self):
        #len(dataset)
        return self.n_samples

class Transmitter():

    def __init__(self,sample_rate,center_freq,gain=-20, sdr_ip="ip:192.168.3.1"):
        self.sdr = adi.Pluto(sdr_ip)
        self.sdr.sample_rate = int(sample_rate)
        self.sdr.tx_rf_bandwidth = int(sample_rate)
        self.sdr.tx_lo = int(center_freq)
        self.sdr.tx_hardwaregain_chan0 = gain
        self.sample_rate = int(sample_rate)
        self.barker_sequence = np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1])

    def transmit(self,bits,sps=1, repeat_tx=True, no_tx_repeats=100,plot=False):
        tx_bits = np.concatenate((self.barker_sequence, bits))
        tx_bits = np.repeat(tx_bits,sps)
        tx_signal = np.array([i+1j*0 for i in tx_bits])
        tx_signal = tx_signal*(2**14)
        if repeat_tx:
            self.sdr.tx_cyclic_buffer = True
            self.sdr.tx(tx_signal)
        else:
            for i in range(no_tx_repeats):
                self.sdr.tx(tx_signal)
        
        if plot:
            plt.figure()
            plt.plot(tx_signal/(2**14))

        



class Receiver():
    
    def __init__(self,sample_rate, center_freq,data_length,sps=20,buffer_size=5000,rx_gain_mode='manual', rx_gain=50.0,sdr_ip = "ip:192.168.3.1"):

        self.sdr = adi.Pluto(sdr_ip)
        self.sdr.rx_lo = int(center_freq)
        self.sdr.rx_rf_bandwidth = int(sample_rate)
        self.sdr.rx_buffer_size=int(buffer_size)
        self.sdr.gain_control_mode_chan0=rx_gain_mode
        self.sdr.rx_hardwaregain_chan0=rx_gain
        self.sps =sps
        self.data_length = data_length
        self.sample_rate = int(sample_rate)
        self.barker_sequence = np.repeat(np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1]),self.sps)
        self.rx_buffer_size = buffer_size
        
    def coarse_freq_sync(self, samples):
        ##works only for BPSK now with s**2 and t/2 in function
        psd = np.fft.fftshift(np.abs(np.fft.fft(samples**2)))
    
        f = np.linspace(-self.sample_rate/2.0, self.sample_rate/2.0, len(psd))
        Ts = 1/self.sample_rate
        offset = f[int(f[np.argmax(psd)])]
        t = np.arange(0, Ts*len(samples), Ts)[:self.rx_buffer_size]
        correct_samples = samples * np.exp(-1j*2*np.pi*offset*t/2.0)

        return correct_samples



    def receive(self,plot=False):

        for i in range(10):
            _ = self.sdr.rx()
        rx_samples = self.sdr.rx()
        

        rx_i = rx_samples.real
        rx_r = self.coarse_freq_sync(rx_i)

        rx_q = rx_samples.imag
        rx_i = self.coarse_freq_sync(rx_q)

        rx = rx_i-rx_q

        if sum(rx)==self.rx_buffer_size or sum(rx)==0:
            print('dc offset error detected')

        corr = np.correlate(rx, self.barker_sequence, 'full')
        
        max_cor_id = np.argmax(np.abs(corr))
        
        rx_bit_frame = rx[max_cor_id+1: max_cor_id+1+(self.sps*self.data_length)]

        res = np.empty((self.data_length,))
        for i in range(self.data_length):
            if np.mean(rx_bit_frame[self.sps*i:(i+1)*self.sps])>=0.5:
                res[i]=1
            else:
                res[i]=0

        if plot:
            plt.figure()
            plt.scatter(np.real(rx_samples), np.imag(rx_samples))
            print(f'Barker detected with max corr value {np.max(corr)/self.sps}')

        if corr[max_cor_id]<0:
            res = (res+1)%2

        
        return res
    

class Modulation():

    def __init__(self,sample_rate,center_freq,gain=-5, sdr_ip="ip:192.168.3.1"):
        self.sdr = adi.Pluto(sdr_ip)
        self.sdr.sample_rate = int(sample_rate)
        self.sdr.tx_rf_bandwidth = int(sample_rate)
        self.sdr.tx_lo = int(center_freq)
        self.sdr.tx_hardwaregain_chan0 = gain
        self.sample_rate = int(sample_rate)
        self.barker_sequence = np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1])

    def OOK(self,bits,freq=20e3,sps=3, repeat_tx=True, no_tx_repeats=100,plot=False):
        tx_bits = np.concatenate((self.barker_sequence, bits))
        tx_bits = np.repeat(tx_bits,sps)

        # self.length_of_data_only = len(bits)*sps
        # N=100
        # t = np.arange(N)/self.sample_rate
        # # print(f'{N/self.sample_rate} secs worth of samples will be transmitted for each bit ')
        # ref_signal = np.exp(2.0j*np.pi*freq*t)

        # tx_signal = np.array([ref_signal*i for i in tx_bits]).flatten()
        tx_signal = np.array([i+1j*0 for i in tx_bits])
        tx_signal = tx_signal*(2**14)
        if repeat_tx:
            self.sdr.tx_cyclic_buffer = True
            self.sdr.tx(tx_signal)
        else:
            for i in range(no_tx_repeats):
                self.sdr.tx(tx_signal)
        
        if plot:
            plt.figure()
            plt.plot(tx_signal/(2**14))
            # print(f'tx samples {tx_signal[:10]}')
        



class Demodulator():
    
    def __init__(self,sample_rate, center_freq,data_length,sps=3,buffer_size=5000,rx_gain_mode='manual', rx_gain=50.0,sdr_ip = "ip:192.168.3.1"):

        self.sdr = adi.Pluto(sdr_ip)
        self.sdr.rx_lo = int(center_freq)
        self.sdr.rx_rf_bandwidth = int(sample_rate)
        self.sdr.rx_buffer_size=int(buffer_size)
        self.sdr.gain_control_mode_chan0=rx_gain_mode
        self.sdr.rx_hardwaregain_chan0=rx_gain
        self.sps =sps
        self.data_length = data_length
        self.sample_rate = int(sample_rate)
        self.barker_sequence = np.repeat(np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1]),self.sps)
        self.rx_buffer_size = buffer_size
        #Clear the buffers
        
    def coarse_freq_sync(self, samples):
        psd = np.fft.fftshift(np.abs(np.fft.fft(samples**2)))
    
        f = np.linspace(-self.sample_rate/2.0, self.sample_rate/2.0, len(psd))
        Ts = 1/self.sample_rate
        offset = f[int(f[np.argmax(psd)])]
        # print(offset)
        t = np.arange(0, Ts*len(samples), Ts)[:self.rx_buffer_size]
        # plt.figure()
        # plt.plot(f, psd)
        # plt.show()
        correct_samples = samples * np.exp(-1j*2*np.pi*offset*t/2.0)

        return correct_samples



    def OOK_demodulate(self,plot=False):

        for i in range(10):
            _ = self.sdr.rx()
        rx_samples = self.sdr.rx()
        # print(len(rx_samples))

        # plt.figure()
        # plt.scatter(np.real(rx_samples), np.imag(rx_samples))

        rx_i = rx_samples.real
        rx_r = self.coarse_freq_sync(rx_i)

        rx_q = rx_samples.imag
        rx_i = self.coarse_freq_sync(rx_q)

        # plt.figure()
        # samples = rx_i+1j*rx_q
        # plt.scatter(np.real(samples), np.imag(samples))

        # rx = rx_i-rx_q - (np.mean(rx_i)+np.mean(rx_q))
        # rx = (rx_i - np.mean(rx_i))-(rx_q-np.mean(rx_q))
        rx = rx_i-rx_q

        
        # rx = np.abs(rx_samples)
        # rx = rx - np.mean(rx)
        # fs = self.sample_rate
        # psd = np.fft.fftshift(np.abs(np.fft.fft(rx**2)))
        # f = np.linspace(-fs/2.0, fs/2.0, len(psd))
        # offset = f[int(f[np.argmax(psd)])]
        # Ts = 1/fs
        # # print(Ts)
        # # print(np.argmax(psd))
        # t = np.arange(0, Ts*len(rx), Ts)[:2000]
        # # print(len(t))
        # rx = rx * np.exp(-1j*2*np.pi*offset*t/2.0)

        # rx_bits = np.where(rx>0,1,0)
        rx_bits = rx
        # print(sum(rx_bits))
        if sum(rx_bits)==self.rx_buffer_size or sum(rx_bits)==0:
            print('dc offset error detected')

        corr = np.correlate(rx_bits, self.barker_sequence, 'full')

        
        # print(f'Barker detected with max corr value {np.max(corr)/self.sps}')
        
        
        max_cor_id = np.argmax(np.abs(corr))
        
        rx_bit_frame = rx_bits[max_cor_id+1: max_cor_id+1+(self.sps*self.data_length)]
        # res = rx_bit_frame[::self.sps]
        # print(f'length after barker {len(rx_bit_frame)}')
        res = np.empty((self.data_length,))
        for i in range(self.data_length):
            # print(np.mean(rx_bit_frame[self.sps*i:(i+1)*self.sps]))
            if np.mean(rx_bit_frame[self.sps*i:(i+1)*self.sps])>=0.5:
                res[i]=1
            else:
                res[i]=0

        if plot:
            # plt.figure()
            # plt.plot(rx_bits[:40])
            plt.figure()
            plt.scatter(np.real(rx_samples), np.imag(rx_samples))
            print(f'Barker detected with max corr value {np.max(corr)/self.sps}')

        if corr[max_cor_id]<0:
            res = (res+1)%2

        
        return res

#----------------------For testing -------------------------------------------------------

class Tx_setting():

    def __init__(self,sample_rate,center_freq,gain=-5, sdr_ip="ip:192.168.3.1"):
        self.sdr = adi.Pluto(sdr_ip)
        self.sdr.sample_rate = int(sample_rate)
        self.sdr.tx_rf_bandwidth = int(sample_rate)
        self.sdr.tx_lo = int(center_freq)
        self.sdr.tx_hardwaregain_chan0 = gain
        self.sample_rate = int(sample_rate)
        self.barker_sequence = np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1])

        # self.barker_sequence = np.array([1,1,1,0,0,0,1,0,0,1,0])

    def Modulation(self,bits,freq=20e3,sps=20, repeat_tx=True, no_tx_repeats=100,plot=False):
        #This setup needs sps to be greater than 10, 20 seems to work well.

        tx_bits = np.concatenate((self.barker_sequence, bits))
        tx_bits = np.repeat(tx_bits,sps)

        #----------Simple modulation--------------------
        tx_signal = np.array([i+1j*0 for i in tx_bits])

        #-----Stores a normalized copy of transmitted points after modulation--------------
        tx_constellation_ref = np.copy(tx_signal)


        tx_signal = tx_signal*(2**14) #Scaling required for pluto hardware (14-bit data converters)

        #---------------Repeated transmission until stopped-------------
        if repeat_tx:
            self.sdr.tx_cyclic_buffer = True
            self.sdr.tx(tx_signal)
        else:
            for i in range(no_tx_repeats):
                self.sdr.tx(tx_signal)
        
        if plot:
            plt.plot(np.real(tx_constellation_ref), np.imag(tx_constellation_ref),'.')
            plt.title('Normalized Tx constellation')
            plt.xlabel('Real axis')
            plt.ylabel('Imaginary axis')
            plt.savefig('constellations/Transmitter_constellation.png')
            
            
        



class Rx_setting():
    #This setup needs sps to be greater than 10, 20 seems to work well.
    
    def __init__(self,sample_rate, center_freq,data_length,sps=20,buffer_size=5000,rx_gain_mode='manual', rx_gain=50.0,sdr_ip = "ip:192.168.3.1"):

        self.sdr = adi.Pluto(sdr_ip)
        self.sdr.rx_lo = int(center_freq)
        self.sdr.rx_rf_bandwidth = int(sample_rate)
        self.sdr.rx_buffer_size=int(buffer_size)
        self.sdr.gain_control_mode_chan0=rx_gain_mode
        self.sdr.rx_hardwaregain_chan0=rx_gain
        self.sps =sps
        self.data_length = data_length
        self.sample_rate = int(sample_rate)
        self.barker_sequence = np.repeat(np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1]),self.sps)
        # self.barker_sequence = np.repeat(np.array([1,1,1,0,0,0,1,0,0,1,0]),self.sps)
        self.rx_buffer_size = buffer_size

        
    def coarse_freq_sync(self, samples):
        # psd = np.fft.fftshift(np.abs(np.fft.fft(samples**2)))
        psd = np.fft.fftshift(np.abs(np.fft.fft(samples*np.conj(samples))))
    
        f = np.linspace(-self.sample_rate/2.0, self.sample_rate/2.0, len(psd))
        Ts = 1/self.sample_rate
        offset = f[int(f[np.argmax(psd)])]

        t = np.arange(0, Ts*len(samples), Ts)
        
        correct_samples = samples * np.exp(-1j*2*np.pi*offset*t/2.0)


        return correct_samples
    

    def data_start_finder(self, data):

        corr = np.correlate(data, self.barker_sequence, 'full')
        max_indxs = np.abs(corr).argsort()[-len(corr):][::-1]
        

        for i in max_indxs:
            if data[i+1: i+1+(self.sps*self.data_length)].size==(self.data_length*self.sps):

                if corr[i]<0:
                    
                    return -data[i+1: i+1+(self.sps*self.data_length)]
                else:
                    
                    return data[i+1: i+1+(self.sps*self.data_length)]
            
        return np.ones(((self.data_length*self.sps),))
            


    def demodulate(self,plot=False):

        #First clear the buffer cache
        for i in range(10):
            _ = self.sdr.rx()

        #raw complex samples at receiver
        rx_samples = self.sdr.rx()

        #-------For some unknown reason this part works the best-----------------
        freq_corrected_samples = self.coarse_freq_sync(rx_samples)

        # rx_i = freq_corrected_samples.real
        # rx_q = rx_samples.imag
        
        # rx = rx_i-rx_q

        rx = (freq_corrected_samples-rx_samples) 
        #---------------------------------------------------------------------

        rx_real = np.copy(rx.real)

        if sum(rx_real)==self.rx_buffer_size or sum(rx_real)==0:
            print('dc offset error detected')

        # corr = np.correlate(rx_real, self.barker_sequence, 'full')

        
        # # print(f'Barker detected with max corr value {np.max(corr)/self.sps}')
        
        
        # max_id = np.argmax(np.abs(corr))
        
        # rx_repeat_symbols = rx_real[max_id+1: max_id+1+(self.sps*self.data_length)]

        
        rx_repeat_symbols = self.data_start_finder(rx_real)

        received_symbols = np.empty((self.data_length,))
        for i in range(self.data_length):

            received_symbols[i] = np.mean(rx_repeat_symbols[self.sps*i:(i+1)*self.sps])

        # res = np.empty((self.data_length,))
        # for i in range(self.data_length):
        #     # print(np.mean(rx_bit_frame[self.sps*i:(i+1)*self.sps]))
        #     if np.mean(rx_repeat_symbols[self.sps*i:(i+1)*self.sps])>=0.5:
        #         res[i]=1
        #     else:
        #         res[i]=0


        
        if plot:
            #----------------Raw received constellation--------------------------
            print('Printing and saving constellations')
            plt.figure()
            plt.plot(np.real(rx_samples), np.imag(rx_samples),'.')
            plt.title('Raw received constellation')
            plt.xlabel('Real axis')
            plt.ylabel('Imaginary axis')
            plt.savefig('constellations/Raw_received_constellation.png')

            #----------------Coarse freq corrected constellation samples--------------------------
            plt.figure()
            plt.plot(np.real(freq_corrected_samples), np.imag(freq_corrected_samples),'.')
            plt.title('Coarse frequency corrected constellation')
            plt.xlabel('Real axis')
            plt.ylabel('Imaginary axis')
            plt.savefig('constellations/Coarse_frequency_corrected_constellation.png')

            #----------------Coarse freq corrected - raw received constellation samples--------------------------
            plt.figure()
            plt.plot(np.real(rx), np.imag(rx),'.')
            plt.title('CFC- RAW samples constellation')
            plt.xlabel('Real axis')
            plt.ylabel('Imaginary axis')
            plt.savefig('constellations/CFC_subtracted_raw_samples_constellation.png')

            #-------------------After barker removal---------------------------------------

            plt.figure()
            plt.plot(np.real(received_symbols), np.imag(received_symbols),'.')
            plt.title('Barker removed data symbols after averaging over repeated symbols ')
            plt.xlabel('Real axis')
            plt.ylabel('Imaginary axis')
            plt.savefig('constellations/Barker_removed_samples_constellation.png')


        # if corr[max_id]<0:
        #     res = (res+1)%2
        else:
            return received_symbols
        

# class Modulation():

#     def __init__(self,sample_rate,center_freq,gain=-5, sdr_ip="ip:192.168.3.1"):
#         self.sdr = adi.Pluto(sdr_ip)
#         self.sdr.sample_rate = int(sample_rate)
#         self.sdr.tx_rf_bandwidth = int(sample_rate)
#         self.sdr.tx_lo = int(center_freq)
#         self.sdr.tx_hardwaregain_chan0 = gain
#         self.sample_rate = int(sample_rate)
#         self.barker_sequence = np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1])

#     def OOK(self,bits,freq=20e3,sps=3, repeat_tx=True, no_tx_repeats=100,plot=False):
#         tx_bits = np.concatenate((self.barker_sequence, bits))
#         tx_bits = np.repeat(tx_bits,sps)

#         # self.length_of_data_only = len(bits)*sps
#         # N=100
#         # t = np.arange(N)/self.sample_rate
#         # # print(f'{N/self.sample_rate} secs worth of samples will be transmitted for each bit ')
#         # ref_signal = np.exp(2.0j*np.pi*freq*t)

#         # tx_signal = np.array([ref_signal*i for i in tx_bits]).flatten()
#         tx_signal = np.array([i+1j*0 for i in tx_bits])

#         if plot:
#             reference = np.copy(tx_signal)
#             # plt.figure()
#             # # plt.plot(tx_signal/(2**14))
#             # plt.scatter(np.real(reference), np.imag(reference))
            
#             # print(f'tx samples {tx_signal[:10]}')
        
#         tx_signal = tx_signal*(2**14)
#         if repeat_tx:
#             self.sdr.tx_cyclic_buffer = True
#             self.sdr.tx(tx_signal)
#         else:
#             for i in range(no_tx_repeats):
#                 self.sdr.tx(tx_signal)

#         if plot:
#             return reference
        
        
        



# class Demodulator():
    
#     def __init__(self,sample_rate, center_freq,data_length,sps=3,buffer_size=5000,rx_gain_mode='manual', rx_gain=50.0,sdr_ip = "ip:192.168.3.1"):

#         self.sdr = adi.Pluto(sdr_ip)
#         self.sdr.rx_lo = int(center_freq)
#         self.sdr.rx_rf_bandwidth = int(sample_rate)
#         self.sdr.rx_buffer_size=int(buffer_size)
#         self.sdr.gain_control_mode_chan0=rx_gain_mode
#         self.sdr.rx_hardwaregain_chan0=rx_gain
#         self.sps =sps
#         self.data_length = data_length
#         self.sample_rate = int(sample_rate)
#         self.barker_sequence = np.repeat(np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1]),self.sps)
#         self.rx_buffer_size = buffer_size
#         #Clear the buffers
        
#     def coarse_freq_sync(self, samples):
#         psd = np.fft.fftshift(np.abs(np.fft.fft(samples**2)))
    
#         f = np.linspace(-self.sample_rate/2.0, self.sample_rate/2.0, len(psd))
#         Ts = 1/self.sample_rate
#         offset = f[int(f[np.argmax(psd)])]
#         # print(offset)
#         t = np.arange(0, Ts*len(samples), Ts)[:self.rx_buffer_size]
#         # plt.figure()
#         # plt.plot(f, psd)
#         # plt.show()
#         correct_samples = samples * np.exp(-1j*2*np.pi*offset*t/2.0)

#         return correct_samples



#     def OOK_demodulate(self,plot=False):

#         for i in range(10):
#             _ = self.sdr.rx()
#         rx_samples = self.sdr.rx()
#         raw_received = np.copy(rx_samples)
#         # print(len(rx_samples))

#         # plt.figure()
#         # plt.scatter(np.real(rx_samples), np.imag(rx_samples))

#         rx_i = rx_samples.real
#         rx_r = self.coarse_freq_sync(rx_i)

#         rx_q = rx_samples.imag
#         rx_i = self.coarse_freq_sync(rx_q)

#         # plt.figure()
#         # samples = rx_i+1j*rx_q
#         # plt.scatter(np.real(samples), np.imag(samples))

#         # rx = rx_i-rx_q - (np.mean(rx_i)+np.mean(rx_q))
#         # rx = (rx_i - np.mean(rx_i))-(rx_q-np.mean(rx_q))
#         rx = rx_i-rx_q

        
#         # rx = np.abs(rx_samples)
#         # rx = rx - np.mean(rx)
#         # fs = self.sample_rate
#         # psd = np.fft.fftshift(np.abs(np.fft.fft(rx**2)))
#         # f = np.linspace(-fs/2.0, fs/2.0, len(psd))
#         # offset = f[int(f[np.argmax(psd)])]
#         # Ts = 1/fs
#         # # print(Ts)
#         # # print(np.argmax(psd))
#         # t = np.arange(0, Ts*len(rx), Ts)[:2000]
#         # # print(len(t))
#         # rx = rx * np.exp(-1j*2*np.pi*offset*t/2.0)

#         # rx_bits = np.where(rx>0,1,0)
#         rx_bits = np.copy(rx)
#         # print(sum(rx_bits))
#         if sum(rx_bits)==self.rx_buffer_size or sum(rx_bits)==0:
#             print('dc offset error detected')

#         corr = np.correlate(rx_bits, self.barker_sequence, 'full')

        
#         # print(f'Barker detected with max corr value {np.max(corr)/self.sps}')
        
        
#         max_cor_id = np.argmax(np.abs(corr))
        
#         rx_bit_frame = rx_bits[max_cor_id+1: max_cor_id+1+(self.sps*self.data_length)]
#         constel_after_barker = np.copy(rx_bit_frame)
#         # res = rx_bit_frame[::self.sps]
#         # print(f'length after barker {len(rx_bit_frame)}')
#         res = np.empty((self.data_length,))
#         for i in range(self.data_length):
#             # print(np.mean(rx_bit_frame[self.sps*i:(i+1)*self.sps]))
#             if np.mean(rx_bit_frame[self.sps*i:(i+1)*self.sps])>=0.5:
#                 res[i]=1
#             else:
#                 res[i]=0

#         # if plot:
#         #     # plt.figure()
#         #     # plt.plot(rx_bits[:40])
#         #     print('----------------Raw received-----------------')
#         #     plt.figure()
#         #     plt.scatter(np.real(raw_received), np.imag(raw_received))

#         #     print('----------------After coarse freq correction-----------------')
#         #     plt.figure()
#         #     plt.scatter(np.real(rx), np.imag(rx))
#         #     print(f'Barker detected with max corr value {np.max(corr)/self.sps}')

#         #     print('----------------After removing barker -----------------')
#         #     plt.figure()
#         #     plt.scatter(np.real(constel_after_barker), np.imag(constel_after_barker))

#         if corr[max_cor_id]<0:
#             res = (res+1)%2

#         if plot:
#             return res,raw_received,rx,constel_after_barker
#         else:
#             return res
        

def rrcosfilter(N, alpha, Ts, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.

    Parameters
    ----------
    N : int
        Length of the filter in samples.

    alpha : float
        Roll off factor (Valid values are [0, 1]).

    Ts : float
        Symbol period in seconds.

    Fs : float
        Sampling Rate in Hz.

    Returns
    ---------

    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.

    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.
    """

    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
        elif alpha != 0 and t == Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif alpha != 0 and t == -Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
                    4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
                    (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)

    return time_idx, h_rrc
