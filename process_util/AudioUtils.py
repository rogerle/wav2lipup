import torchaudio


class AudioUtils():

    def load_wav(self,path, sr=sr):
        wavform,fs = torchaudio.load(path)