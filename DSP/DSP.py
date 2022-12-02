import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np

class DSP(object):
    def __init__(self, path=None, epsiBF=1e-3,epsiSVE=1e-3,sr=8000, n_fft=256,n_sensor=7):
        if path is None : 
            self.lib = ctypes.cdll.LoadLibrary("build/libDSP.so")
        else :
            self.lib = ctypes.cdll.LoadLibrary(path)
        _doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')

        self.epsiBF = epsiBF
        self.epsiSVE = epsiSVE

    #print(dir(lib))

        self.lib.DSP_new.argtypes = [ctypes.c_double,ctypes.c_double]
        self.lib.DSP_new.restype = ctypes.c_void_p

        self.lib.DSP_Process.argtypes = [ctypes.c_void_p,_doublepp , ctypes.c_int] 
        self.lib.DSP_Process.restype = ctypes.c_void_p

        self.lib.DSP_reset.argtypes = [ctypes.c_double,ctypes.c_double]
        self.lib.DSP_reset.restype = ctypes.c_void_p

        self.obj = self.lib.DSP_new(epsiBF,epsiSVE)

        print("INFO::DSP::loaded")

    """
        data = np.arange(25).reshape((5, 5)) 
        arr=np.array(data,dtype=np.float64)
    """
    def Process(self,x,len_data):
        xpp = (x.__array_interface__['data'][0] + np.arange(x.shape[0])*x.strides[0]).astype(np.uintp) 
        self.lib.DSP_Process(self.obj,xpp,len_data)

    def Reset(self):
        self.obj = self.lib.DSP_reset(self.epsiBF,self.epsiSVE)

## for debugging
if __name__ == "__main__":
    import librosa as rs
    import soundfile as sf


    dsp = DSP()
    x = rs.load("10.wav",sr=8000,mono=False)[0]
    x = x.astype(np.float64)
    x = np.ascontiguousarray(x)

    print(np.mean(x))
    len_data = int(x.shape[1]/64)

    dsp.Process(x,len_data)
    print(np.mean(x))

    sf.write("10_estim.wav",x[0],8000)
