import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
np.seterr(invalid='ignore')  # 忽视无效浮点运算的
# 第一个类
class CSignal(object):
    def __init__(self, Fs, vecX):
        self.Fs = Fs
        self.vecX = vecX

    def CopySelf(self):
        new_vecX = self.vecX
        new_Fs = self.Fs
        new_sig = CSignal(new_Fs, new_vecX)
        return new_sig

    def DispWave(self):
        t = np.array([i for i in range(len(self.vecX))])/self.Fs
        plt.plot(t, self.vecX)
        plt.show()

    def GetModulus(self):
        return np.sum(self.vecX * self.vecX)\
                      / len(self.vecX)

    def GetVecX(self):
        vecX = self.vecX
        return vecX

    def GetFs(self):
        Fs = self.Fs
        return Fs

    def CycleShift(self, nLag):
        n = len(self.vecX)
        idxTail = [i for i in range(n-nLag, n)]
        idxHead = [i for i in range(0, n-nLag)]
        vecX = np.append(self.vecX[idxTail], self.vecX[idxHead])
        sig = CSignal(self.Fs, vecX)
        return sig

    def DFT(self):
        n = len(self.vecX)
        idx = int(n/2)
        vec = np.array([i for i in range(n)])
        m1 = np.transpose([vec]) * vec
        m = np.exp(-1j*2*np.pi/n*m1)
        vecCK1 = (m.dot(np.transpose([self.vecX]))/n).transpose()  # 列转行
        vecCK2 = np.abs(vecCK1).tolist()
        vecCK = []
        for i in vecCK2:
            for j in i:
                vecCK.append(j)
        # vecCK = vecCK[0:idx+1]
        deltaF = self.Fs/(n-1)
        vecF = np.array([i for i in range(n)])*deltaF
        # vecF = vecF[0:idx+1]
        return vecCK, vecF

    def myFFT(self):
        n = len(self.vecX)
        vecCk = np.abs(fft(self.vecX))/n
        deltaF = self.Fs / (n - 1)
        vecF = np.array([i for i in range(n)]) * deltaF
        return vecCk, vecF

    def DFTMatrix(self):
        n = len(self.vecX)
        if n > 8:
            print('Points not more than 8')
        else:
            vec = np.array([i for i in range(n)])
            m1 = np.transpose([vec]) * vec
            m = np.exp(-1j * 2 * np.pi / n * m1)
            print(m)

    def AutoSpectrum(self, tFrame, tShift):
        vecX = self.vecX
        Fs = self.Fs
        nFrame = int(tFrame*Fs)
        index = int(nFrame/2)
        # index = np.array([i for i in range(1, a+1)])
        nShift = int(tShift*Fs)
        n = nFrame
        deltaF = 1/tFrame
        vecF = np.array([i for i in range(n)])*deltaF
        vecF = vecF[0:index]
        b = len(vecX)-nFrame+1
        idxStart = np.array([i for i in range(1, b+1, nShift)])
        nFrameNum = len(idxStart)
        GxxTmp = np.array([0 for i in range(nFrame-1)])
        for ind in range(0, nFrameNum-1):
            nStart = idxStart[ind]
            nEnd = nStart + nFrame-1
            xFrame = vecX[nStart:nEnd]
            xkFrame = fft(xFrame)/nFrame
            c = xkFrame*xkFrame.conjugate()
            GxxTmp = GxxTmp + c
        GxxTmp = 4*(GxxTmp/nFrameNum)
        Gxx = GxxTmp[0:index]
        Gxx = np.abs(Gxx)
        # Gxx = np.sqrt(Gxx)
        return Gxx, vecF

    def STFT(self, tFrame, tShift):
        vecX = self.vecX
        Fs = self.Fs
        nFrame = int(tFrame * Fs)
        index = int(nFrame / 2)
        nShift = int(tShift * Fs)
        n = nFrame
        deltaF = 1 / tFrame
        vecF = np.array([i for i in range(n)]) * deltaF
        vecF = vecF[0:index]
        b = len(vecX) - nFrame + 1
        idxStart = np.array([i for i in range(1, b + 1, nShift)])
        nFrameNum = len(idxStart)
        vecT = np.array([i for i in range(0, nFrameNum)])*tShift
        matxGxx = np.zeros((len(vecF), nFrameNum))
        for ind in range(0, nFrameNum):
            nStart = idxStart[ind]
            nEnd = nStart + nFrame - 1
            xFrame = vecX[nStart:nEnd]
            xkFrame = fft(xFrame) / nFrame
            c = xkFrame * xkFrame.conjugate()
            c =c.transpose()
            c = np.abs(c)
            c = c[0:index]
            matxGxx[:, ind] = c
        return matxGxx, vecF, vecT

# 下面是CSignal的子类们，所有子类返回的均是父类的实例化对象
class CSquareSignal(CSignal):
    def __init__(self, Fs, vecX,  dA, dF, dP, tLen):
        super().__init__(Fs, vecX)
        self.dA = dA
        self.dF = dF
        self.dP = dP
        self.tLen = tLen

    def SquareSignal(self):
        # 第一种：用外部输入的 余弦向量
        # vecX = self.dA*np.array([1 for i in range(len(self.vecX))])
        # for i in range(int(len(self.vecX))):
        #     if self.vecX[i] > 0:
        #         vecX[i] = self.dA
        #     else:
        #         vecX[i] = self.dA * (-1)
        # 第二种：用方法内部的 余弦向量，此时外部调用是第二个参数为：None
        vecT = np.arange(0, self.tLen, 1 / self.Fs)
        vecXTmp = np.cos(2*np.pi*self.dF*vecT + self.dP)
        vecX = self.dA * np.array([1 for i in range(len(vecXTmp))])
        for i in range(len(vecXTmp)):
            if vecXTmp[i] < 0:
                vecX[i] = vecX[i]*(-1)
        sig_square = CSignal(self.Fs, vecX)
        return sig_square #返回一个父类的实例化对象
        # # 第三种 在子类方法中调用父类方法
        # vecT = np.arange(0, self.tLen, 1 / self.Fs)
        # vecXTmp = np.cos(2 * np.pi * self.dF * vecT + self.dP)
        # vecX = self.dA * np.array([1 for i in range(len(vecXTmp))])
        # for i in range(len(vecXTmp)):
        #     if vecXTmp[i] < 0:
        #         vecX[i] = vecX[i]*(-1)
        # self.vecX = vecX
        # super().DispWave()

class CNoiseSignal(CSignal):
    def __init__(self, Fs, vecX, dA, tLen):
        super().__init__(Fs, vecX)
        self.dA = dA
        self.tLen = tLen

    def NoiseSignal(self):
        vecT = np.arange(0, self.tLen, 1/self.Fs)
        vecXTmp = self.dA * np.random.randn(1, int(len(vecT)))
        vecX = vecXTmp[0][:]
        sig_Noise = CSignal(self.Fs, vecX)
        return sig_Noise

class CHarmonicSignal(CSignal):
    def __init__(self, Fs, vecX, dA, dF, dP, tLen):
        super().__init__(Fs, vecX)
        self.dA = dA
        self.dF = dF
        self.dP = dP
        self.tLen = tLen

    def HarmonicSignal(self):
        vecT = np.array([i for i in np.arange(0, self.tLen, 1/self.Fs)])
        vecX = self.dA * np.cos(2*np.pi*self.dF*vecT + self.dP)
        sig_harmonic = CSignal(self.Fs, vecX)
        return sig_harmonic

class CCplxPeriodSignal(CSignal):
    def __init__(self, Fs, vecX, vecA, vecF, vecP, tLen):
        super().__init__(Fs, vecX)
        self.vecA = vecA
        self.vecF = vecF
        self.vecP = vecP
        self.tLen = tLen

    def CplxPeriodSignal(self):
        vecT = np.array([i for i in np.arange(0, self.tLen, 1 / self.Fs)])
        a = np.transpose([np.array(self.vecA)]) * np.array([1.0 for i in range(int(len(vecT)))])
        b = np.transpose([np.array(self.vecF)]) * vecT
        c = np.transpose([np.array(self.vecP)]) * np.array([1.0 for i in range(int(len(vecT)))])
        m = a * np.cos(2*np.pi*b + c)
        vecX = m.sum(axis=0)
        sig_CPlxPeriod = CSignal(self.Fs, vecX)
        return sig_CPlxPeriod

# 在类的外部定义相关系数的算法函数，类里边的方法直接调用即可
def CaclcCorrelationCoeff(vecX1,vecX2):
    a = vecX1-np.mean(vecX1)
    b = np.transpose([vecX2-np.mean(vecX2)])
    c = vecX2-np.mean(vecX2)
    d = np.transpose([vecX1-np.mean(vecX1)])
    dCorr = (a.dot(b)) / np.sqrt((a.dot(d))*(c.dot(b)))
    dCorr = float(dCorr)
    return dCorr

#第二个类
class CSignalPair(object):
    def __init__(self, sig1, sig2):
        self.sig1 = sig1
        self.sig2 = sig2

    def Add(self):
        vecX1 = self.sig1.GetVecX()
        vecX2 = self.sig2.GetVecX()
        Fs = self.sig1.GetFs()
        vecX = vecX1+vecX2
        sig_Add = CSignal(Fs, vecX)
        return sig_Add

    def CorrelationCoeff(self):
        vecX1 = self.sig1.GetVecX()
        vecX2 = self.sig2.GetVecX()
        corr = CaclcCorrelationCoeff(vecX1, vecX2)
        return corr

    def CrossSpectrum(self, tFrame, tShift):
        vecX1 = self.sig1.GetVecX()
        vecX2 = self.sig2.GetVecX()
        Fs = self.sig1.GetFs()
        nFrame = int(tFrame*Fs)
        index = int(nFrame/2)
        # index = np.array([i for i in range(1, a+1)])
        nShift = int(tShift*Fs)
        n = nFrame
        deltaF = 1/tFrame
        vecF = np.array([i for i in range(n)])*deltaF
        vecF = vecF[0:index]
        b = len(vecX1)-nFrame+1
        idxStart = np.array([i for i in range(1, b+1, nShift)])
        nFrameNum = len(idxStart)
        GxyTmp = np.array([0 for i in range(nFrame-1)])
        for ind in range(0, nFrameNum-1):
            nStart = idxStart[ind]
            nEnd = nStart + nFrame-1
            # idx = [i for i in range(nStart, nEnd+1)]
            xFrame = vecX1[nStart:nEnd]
            yFrame = vecX2[nStart:nEnd]
            xkFrame = fft(xFrame)/nFrame
            ykFrame = fft(yFrame)/nFrame
            c = xkFrame*ykFrame.conjugate()
            GxyTmp = GxyTmp + c
        GxyTmp = 4*(GxyTmp/nFrameNum)
        Gxy = GxyTmp[0:index]
        # Gxy = np.abs(Gxy)
        # Gxy = np.sqrt(Gxy)
        return Gxy, vecF

    def FreqResponse1(self, tFrame, tShift):
        sigy = self.sig2.GetVecX()
        Fs = self.sig2.GetFs()
        Gyy, vecF = CSignal(Fs, sigy).AutoSpectrum(tFrame, tShift)
        Gxy, vecF = self.CrossSpectrum(tFrame, tShift)
        H1 = Gyy/Gxy  # 一维数组之间 / 实现对应项除
        return H1, vecF

    def FreqResponse2(self, tFrame, tShift):
        sigx = self.sig1.GetVecX()
        Fs = self.sig1.GetFs()
        Gxx, vecF = CSignal(Fs, sigx).AutoSpectrum(tFrame, tShift)  # 跨类之间方法调用，先生成类的实例化对象，在调用方法
        Gyx, vecF = self.CrossSpectrum(tFrame, tShift)  # 同类之间方法调用， self.方法名即可
        H2 = Gyx/Gxx  # 一维数组之间 / 实现对应项除
        return H2, vecF

    def FreqResponse3(self, tFrame, tShift):
        sigy = self.sig2.GetVecX()
        Fs = self.sig2.GetFs()
        Gyy, vecF = CSignal(Fs, sigy).AutoSpectrum(tFrame, tShift)
        Gxy, vecF = self.CrossSpectrum(tFrame, tShift)
        H1 = Gyy / Gxy  # 一维数组之间 / 实现对应项除
        sigx = self.sig1.GetVecX()
        Fs = self.sig1.GetFs()
        Gxx, vecF = CSignal(Fs, sigx).AutoSpectrum(tFrame, tShift)
        Gyx, vecF = self.CrossSpectrum(tFrame, tShift)
        H2 = Gyx / Gxx  # 一维数组之间 / 实现对应项除
        H3 = np.sqrt(H1*H2)  # 一维数组之间 * 实现对应项相除
        return H3, vecF

    def Coherence(self, tFrame, tShift):
        Gxx, vecF = self.sig1.AutoSpectrum(tFrame, tShift)
        Gyy, vecF = self.sig2.AutoSpectrum(tFrame, tShift)
        Gxy, vecF = self.CrossSpectrum(tFrame, tShift)
        Gxy = np.abs(Gxy)
        Gxy = np.power(Gxy, 2)
        a = Gxx*Gyy
        vecGama = Gxy/a
        return vecGama, vecF


    def CycleCorrelationFunc(self):
        vecX1 = self.sig1.GetVecX()
        vecX2 = self.sig2.GetVecX()
        n = len(vecX2)
        vecCorr = np.array([])
        vecleg = np.array([i for i in range(-n+1, n+1)])
        # dCorr = self.CorrelationCoeff()
        dCorr = CaclcCorrelationCoeff(vecX1, vecX2)
        vecCorr = np.append(vecCorr, dCorr)
        for i in range(1, len(vecleg)):
            self.sig2 = self.sig2.CycleShift(1)
            # dCorr = self.CorrelationCoeff()
            vecX2 = self.sig2.GetVecX()
            dCorr = CaclcCorrelationCoeff(vecX1, vecX2)
            vecCorr = np.append(vecCorr, dCorr)
        return vecCorr, vecleg

    def CorrelationFunc(self):
        vecX1 = self.sig1.GetVecX()
        vecX2 = self.sig2.GetVecX()
        n1 = len(vecX1)
        n2 = len(vecX2)
        vecleg = np.array([i for i in range(-n2+1, n1)])
        vecCorr = np.array([])
        vec1 = np.append(np.append(np.array([0 for i in range(n2-1)]), vecX1), np.array([0 for i in range(n2-1)]))
        vec2 = np.append(vecX2, np.array([0 for i in range(n1+n2-2)]))
        vec = vec1*vec2
        bool = np.logical_not(vec==0)
        corr = CaclcCorrelationCoeff(vec1[bool], vec2[bool])
        vecCorr = np.append(vecCorr, corr)
        for i in range(1, len(vecleg)):
            vec2 = np.append(np.append(np.array([0 for i in range(i)]), vecX2), np.array([0 for i in range(n1+n2-2-i)]))
            vec = vec1*vec2
            bool =  np.logical_not(vec==0)
            corr = CaclcCorrelationCoeff(vec1[bool], vec2[bool])
            vecCorr = np.append(vecCorr, corr)
        return vecCorr, vecleg


