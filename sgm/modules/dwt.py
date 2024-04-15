import pywt
import ptwt
import torch.nn as nn

class DWT2D(nn.Module):
    def __init__(self, kernel='haar', **kwargs):
        super(DWT2D, self).__init__(**kwargs)
        self.kernel = kernel

    def decompose(self, x):
        coeffs = ptwt.wavedec2(x, pywt.Wavelet(self.kernel),
                                level=1, mode="symmetric")
        cA, (cH, cV, cD) = coeffs
        return cA, cH, cV, cD

    def inverse(self, coff):
        return ptwt.waverec2(coff, pywt.Wavelet(self.kernel))