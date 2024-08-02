import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch
import random
from tqdm import tqdm
from models.whisper.model import Whisper, ModelDimensions
from models.whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram


def pred_melspec16(wavPath, ppgPath):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppgln = 2*(audln // 320) #in order to match spectrogram shape it must truncate at number frames of 320 hop, so 2*frames(hop160)
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).data.cpu().float().numpy()
#     with torch.no_grad():
#         ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
#         ppg = ppg[:ppgln,]  # [length, dim=1280]
#         # print(ppg.shape)
    np.save(ppgPath, mel[:,:ppgln], allow_pickle=False)


# +
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-m", "--melspec", help="melspec", dest="melspec", required=True)
    args = parser.parse_args()
    print(args.wav)
    print(args.melspec)

    os.makedirs(args.melspec, exist_ok=True)
    wavPath = args.wav
    melspecPath = args.melspec

#     whisper = load_model(os.path.join("whisper_pretrain", "large-v2.pt"))
    spkPaths = os.listdir(wavPath)
    random.shuffle(spkPaths)

    for spks in spkPaths:
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{melspecPath}/{spks}", exist_ok=True)

            files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]
            for file in tqdm(files, desc=f'Processing melspecs {spks}'):
                if file.endswith(".wav"):
                    # print(file)
                    file = file[:-4]
                    path_wav = f"{wavPath}/{spks}/{file}.wav"
                    path_ppg = f"{melspecPath}/{spks}/{file}.m16"
                    if os.path.isfile(f"{melspecPath}.npy"):
                        continue
                    pred_melspec16(path_wav, path_ppg)
