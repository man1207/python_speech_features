import numpy as np
import scipy.io.wavfile as wav

class LowcFE():

    PITCH_MIN = 40  # minimum allowed pitch, 200 Hz
    PITCH_MAX = 120  # maximum allowed pitch, 66 Hz
    PITCHDIFF = PITCH_MAX - PITCH_MIN
    POVERLAPMAX = PITCH_MAX >> 2  # maximum pitch OLA window
    HISTORYLEN = PITCH_MAX * 3 + POVERLAPMAX  # history buffer length
    NDEC = 2  # 2:1 decimation
    CORRLEN = 160  # 20 ms correlation length
    CORRBUFFLEN = CORRLEN + PITCH_MAX  # correlation buffer length
    CORRMINPOWER = 250.0  # minimum power
    EOVERLAPINCR = 32  # end OLA increment per frame, 4 ms
    FRAMESZ = 80  # 10 ms at 8000 Hz
    ATTENFAC = .2  # attenuation factor per 10 ms frame
    ATTENINCR = ATTENFAC/FRAMESZ  # attenuation per sample

    def __init__(self):
        
        self.erasecnt = 0  # consecutive erased frames
        self.poverlap = LowcFE.POVERLAPMAX  # overlap based on pitch
        self.poffset = 0  # offset into pitch period
        self.pitch = 0  # pitch estimate
        self.pitchblen = 0  # current pitch buffer length
        self.pitchbufend = LowcFE.HISTORYLEN  # end of pitch buffer
        self.pitchbufstart = 0  # start of pitch buffer
        self.pitchbuf = np.zeros(LowcFE.HISTORYLEN)  # buffer for cycles of speech
        self.lastq = np.zeros(LowcFE.POVERLAPMAX)  # saved last quarter wavelength
        self.history = np.zeros(LowcFE.HISTORYLEN) # history buffer

    def overlapadd(self, l, r, s, cnt):
        incr = 1.0/cnt
        lw = 1.0 - incr
        rw = incr
        for i in range(cnt):
            t = lw*l[i] + rw*r[i]
            if t > 32767.0:
                t = 32767.0
            elif t < -32768.0:
                t = -32768.0
            s[i] = t
            lw -= incr
            rw += incr

    def dofe(self):  # synthesize speech for erasure
        s = np.zeros(LowcFE.FRAMESZ)
        if self.erasecnt == 0:
            self.pitchbuf = self.history.copy()
            self.pitch = self.findpitch()
            self.poverlap = self.pitch >> 2
            self.lastq = self.pitchbuf[self.pitchbufend - self.poverlap:].copy()
            self.poffset = 0
            self.pitchblen = self.pitch
            self.pitchbufstart = self.pitchbufend - self.pitchblen
            self.overlapadd(self.lastq, self.pitchbuf[self.pitchbufstart - self.poverlap:self.pitchbufstart],
                            self.pitchbuf[self.pitchbufend - self.poverlap:], self.poverlap)
            self.history[LowcFE.HISTORYLEN - self.poverlap:] = self.pitchbuf[self.pitchbufend - self.poverlap:].copy()
            self.getfespeech(s, LowcFE.FRAMESZ)
        elif self.erasecnt == 1 or self.erasecnt == 2:
            tmp = np.zeros(LowcFE.POVERLAPMAX)
            saveoffset = self.poffset  # save offset for OLA
            self.getfespeech(tmp, self.poverlap)  # continue with old pitchbuf
            self.poffset = saveoffset
            while self.poffset > self.pitch:
                self.poffset -= self.pitch
            self.pitchblen += self.pitch  # add a period
            self.pitchbufstart = self.pitchbufend - self.pitchblen
            self.overlapadd(self.lastq, self.pitchbuf[self.pitchbufstart - self.poverlap:self.pitchbufstart],
                            self.pitchbuf[self.pitchbufend - self.poverlap:], self.poverlap)  # overlap add old pitchbuffer with new
            self.getfespeech(s, LowcFE.FRAMESZ)
            self.overlapadd(tmp, s, s, self.poverlap)
            self.scalespeech(s)
        elif self.erasecnt > 5:
            pass
        else:
            self.getfespeech(s, LowcFE.FRAMESZ)
            self.scalespeech(s)
        self.erasecnt += 1
        return self.__savespeech(s)

    def addtohistory(self, s):  # add a good frame to history buffer
        if (self.erasecnt):
            overlapbuf = np.zeros(LowcFE.FRAMESZ)
            olen = self.poverlap + (self.erasecnt - 1) * LowcFE.EOVERLAPINCR
            olen = LowcFE.FRAMESZ if olen > LowcFE.FRAMESZ else olen
            self.getfespeech(overlapbuf, olen)
            self.overlapaddatend(s, overlapbuf, olen)
            self.erasecnt = 0
        return self.__savespeech(s)

    def scalespeech(self, s):
        g = 1 - (self.erasecnt - 1) * LowcFE.ATTENFAC
        for i in range(LowcFE.FRAMESZ):
            s[i] = s[i] * g
            g -= LowcFE.ATTENINCR

    def getfespeech(self, s, sz):
        j = 0
        while(sz):
            cnt = self.pitchblen - self.poffset
            if (cnt > sz):
                cnt = sz
                # print()
            for i in range(cnt):
                s[i + j] = self.pitchbuf[self.pitchbufstart + self.poffset + i]
            self.poffset += cnt
            if self.poffset == self.pitchblen:
                self.poffset = 0
            sz -= cnt
            j += cnt

    def findpitch(self):
        # coarse search
        l = self.pitchbufend - LowcFE.CORRLEN
        r = self.pitchbufend - LowcFE.CORRBUFFLEN
        rp = r
        energy = 0
        corr = 0
        pitch = 0
        for i in range(LowcFE.CORRLEN):
            if i%LowcFE.NDEC == 0:
                energy += self.pitchbuf[rp+i]*self.pitchbuf[rp+i]
                corr += self.pitchbuf[rp+i]*self.pitchbuf[l+i]
        scale = energy
        if scale < LowcFE.CORRMINPOWER:
            scale = LowcFE.CORRMINPOWER
        corr /= np.sqrt(scale)
        bestcorr = corr
        bestmatch = 0
        for j in range(LowcFE.NDEC, LowcFE.PITCHDIFF + 1):
            if j%LowcFE.NDEC == 0:
                energy -= self.pitchbuf[rp]*self.pitchbuf[rp]
                energy += self.pitchbuf[rp+LowcFE.CORRLEN]*self.pitchbuf[rp+LowcFE.CORRLEN]
                rp += LowcFE.NDEC
                corr = 0
                for i in range(LowcFE.CORRLEN):
                    if i%LowcFE.NDEC == 0:
                        corr += self.pitchbuf[rp+i]*self.pitchbuf[l+i]
                scale = energy
                if scale < LowcFE.CORRMINPOWER:
                    scale = LowcFE.CORRMINPOWER
                corr /= np.sqrt(scale)
                if corr >= bestcorr:
                    bestcorr = corr
                    bestmatch = j
        # fine search
        j = bestmatch - (LowcFE.NDEC - 1)
        if j < 0:
            j = 0
        k = bestmatch + (LowcFE.NDEC - 1)
        if k > LowcFE.PITCHDIFF:
            k = LowcFE.PITCHDIFF
        rp = r+j
        energy = 0
        corr = 0
        for i in range(LowcFE.CORRLEN):
            energy += self.pitchbuf[rp+i]*self.pitchbuf[rp+i]
            corr += self.pitchbuf[rp+i]*self.pitchbuf[l+i]
        scale = energy
        if scale < LowcFE.CORRMINPOWER:
            scale = LowcFE.CORRMINPOWER
        corr /= np.sqrt(scale)
        bestcorr = corr
        bestmatch = j
        for j in (j+1, k+1):
            energy -= self.pitchbuf[rp]*self.pitchbuf[rp]
            energy += self.pitchbuf[rp+LowcFE.CORRLEN]*self.pitchbuf[rp+LowcFE.CORRLEN]
            rp += 1
            corr = 0
            for i in range(LowcFE.CORRLEN):
                corr += self.pitchbuf[rp+i]*self.pitchbuf[l+i]
            scale = energy
            if scale < LowcFE.CORRMINPOWER:
                scale = LowcFE.CORRMINPOWER
            corr /= np.sqrt(scale)
            if corr >= bestcorr:
                bestcorr = corr
                bestmatch = j

        return LowcFE.PITCH_MAX - bestmatch

    def overlapaddatend(self, s, f, cnt):
        incr = 1.0/cnt
        gain = 1.0 - (self.erasecnt - 1)*LowcFE.ATTENFAC
        if gain < 0.:
            gain = 0.
        incrg = incr*gain
        lw = (1.0 - incr)*gain
        rw = incr
        for i in range(cnt):
            t = lw*f[i] + rw*s[i]
            if t > 32767.0:
                t = 32767.0
            elif t < -32768.0:
                t = -32768.0
            s[i] = t
            lw -= incrg
            rw += incr

    def __savespeech(self, s):
        historyend = self.history[LowcFE.FRAMESZ:]
        for i in range(LowcFE.HISTORYLEN - LowcFE.FRAMESZ):
            self.history[i] = historyend[i]
        for i in range(LowcFE.HISTORYLEN - LowcFE.FRAMESZ, LowcFE.HISTORYLEN):
            self.history[i] = s[i + LowcFE.FRAMESZ - LowcFE.HISTORYLEN]
        return self.history[LowcFE.HISTORYLEN - LowcFE.FRAMESZ - LowcFE.POVERLAPMAX:LowcFE.HISTORYLEN - LowcFE.POVERLAPMAX]

    def run(self, signal, loss_label):
        num_frames = len(signal) // LowcFE.FRAMESZ
        output_data = []
        for i in range(num_frames):
            packet = signal[i*LowcFE.FRAMESZ: (i+1)*LowcFE.FRAMESZ]
            packet_good = not loss_label[i]
            if packet_good:
                s = self.addtohistory(packet.copy())
            else:
                s = self.dofe()
            output_data.extend(s)
        output_data = np.array(output_data[30:], dtype=signal.dtype)

        return output_data


class PacketLossSimulator:
    def __init__(self, pN, pL, frame_size=80 ):
        self.pL = pL  # вероятность остаться в состоянии Loss
        self.pN = pN  # вероятность остаться в состоянии Non-Loss
        self.loss_state = False
        self.frame_size = frame_size
        self.loss_label = []
        self.plr = 0

    def simulate(self, data, sample_rate):
        num_frames = len(data)
        self.loss_label = np.zeros(num_frames, dtype=bool)

        for i in range(num_frames):
            if self.loss_state:
                if np.random.rand() < self.pL:
                    data[i*self.frame_size:(i+1)*self.frame_size] = 0
                    self.loss_label[i] = True
                else:
                    self.loss_state = False
            else:
                if np.random.rand() < (1 - self.pN):
                    self.loss_state = True
                    data[i*self.frame_size:(i+1)*self.frame_size] = 0
                    self.loss_label[i] = True

        return data, self.loss_label

    def calculate_packet_loss_rate(self):
        total_packets = len(self.loss_label)
        lost_packets = np.sum(self.loss_label)
        self.plr = lost_packets / total_packets
        return self.plr
