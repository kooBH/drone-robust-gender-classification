import numpy as np


class BufferManager() : 
    def __init__(self,
    n_channel = 7, 
    n_chunk=1024
    ,n_unit=4,
    max_unit=200,
    pre_unit=4,
    log=False
    ):

        self.n_channel = n_channel
        self.n_chunk = n_chunk
        self.n_unit = n_unit
        self.max_unit = max_unit

        self.idx = 0
        self.sz_unit = self.n_unit * self.n_chunk

        self.idx_pre = 0
        self.pre_unit=pre_unit

        self.flag_utt = False    
        self.idx_utt = 0
        self.flag_ret_utt = False

        self.buffer = np.zeros((n_channel,self.sz_unit))
        self.utterance = np.zeros((n_channel,max_unit*n_chunk*n_unit))
        self.log = log

    def feed(self,chunk):
        #print("BufferManager::idx {}".format(self.idx))
        self.buffer[:,self.idx*self.n_chunk:(self.idx+1)*self.n_chunk] = chunk

        self.idx+=1 
        if self.idx < self.n_unit :
            return None, self.flag_ret_utt
        # unit
        else :
            #self.buffer[:,0:-self.sz_unit] = self.buffer[:,self.sz_unit:]
            #self.buffer[:,self.n_unit-self.sz_unit:] = 0
            self.idx = 0 

            #print("{} | utt {} | pre {}".format(self.flag_utt,self.idx_utt,self.idx_pre))

            # stacking utterance
            if self.flag_utt :

                self.utterance[:,(self.idx_utt+self.idx_pre)*self.sz_unit:(self.idx_utt+self.idx_pre+1)*self.sz_unit] = self.buffer
                self.idx_utt +=1

                if self.idx_utt == self.max_unit-1 : 
                    print("BufferManager::Utterance is too long. Forced prossesing")
                    self.flag_ret_utt=True

            # manage shift buffer for pre unit
            else : 
                if self.idx_pre == self.pre_unit :
                    # shift
                    self.utterance[:,0:(self.idx_pre-1)*self.sz_unit]=self.utterance[:,self.sz_unit:self.idx_pre * self.sz_unit]
                    # append
                    self.utterance[:,(self.idx_pre-1)*self.sz_unit:self.idx_pre*self.sz_unit] =self.buffer
                else :
                    # insert
                    self.utterance[:,(self.idx_pre)*self.sz_unit:(self.idx_pre+1)*self.sz_unit] =self.buffer
                    self.idx_pre +=1
                    
            return self.buffer, self.flag_ret_utt

    def toggle_utt(self,flag):
        if self.flag_utt == flag : 
            return False
        if self.flag_utt : 
            if self.log :
                print("BufferManager::Utterance Ends")
            self.flag_utt = False
            self.flag_ret_utt = True
        else : 
            if self.log :
                print("BufferManager::Utterance Starts")
            self.flag_utt = True
            return True
        return False
        

    def get_utterance(self):
        self.flag_ret_utt = False
        ret_data = np.copy(self.utterance[:,:(self.idx_utt+self.idx_pre)*self.sz_unit])
        self.idx_utt=0
        self.idx_pre = 0
        self.utterance[:,:]=0
        return ret_data
