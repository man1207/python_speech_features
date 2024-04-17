from base import logpow, stft, ilogpow

class DnnPLC:
    
    def __init__(self, model, n_feature)
        self.model = model
        self.error_count = 0
        self.error = False
        self.frame_size = 80
        self.window_size = 160
        self.history_num_frames = n_feature+1
        self.sample_rate = 8000
        self.history_buffer = np.zeros(self.history_num_frames * self.frame_size) # history buffer

    def dofe(self):
        pass

    def addtohistory(self, s):
        if (self.error):
            s = self.__predict_frame()
            self.error = False
        return self.__savespeech(s)

    def __predict_frame(self, s):
        # берем history_buffer
        magnitude, angle = stft(self.history_buffer, 
                    samplerate=self.sample_rate, 
                    winlen=self.window_size/self.sample_rate, 
                    winstep=self.frame_size/self.sample_rate, 
                    nfft=self.window_size,
                    winfunc = np.hamming)
        
        feat_phase = np.stack([np.sin(angle), np.cos(angle)], axis=-1)
        feat_logpow = self.__logpow(magnitude)

        logpow = self.__model_logpow.predict(feat_logpow)
        phase = self.__model_phase.predict(feat_phase)
        
        
        return f

    def __logpow(signal):
        pspec = np.maximum(signal**2, 1e-12)
        return np.log10(pspec) 

    def __savespeech(self, s):
        self.history_buffer[:-self.frame_size] = self.history_buffer[self.frame_size:]
        self.history_buffer[-self.frame_size:] = s[:self.frame_size]
        return self.history_buffer[-self.frame_size:]
