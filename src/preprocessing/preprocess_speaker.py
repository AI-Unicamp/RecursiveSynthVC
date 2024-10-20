import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import argparse

from tqdm import tqdm
from functools import partial
from argparse import RawTextHelpFormatter
from multiprocessing.pool import ThreadPool

from src.models.encoder.speaker.models import LSTMSpeakerEncoder
from src.models.encoder.speaker.config import SpeakerEncoderConfig
from src.models.encoder.speaker.utils.audio import AudioProcessor
from src.models.encoder.speaker.infer import read_json

def process_wav_speaker(wav_file, output_path, use_cuda, speaker_encoder_ap, speaker_encoder):
    waveform = speaker_encoder_ap.load_wav(
        wav_file, sr=speaker_encoder_ap.sample_rate
    )
    spec = speaker_encoder_ap.melspectrogram(waveform)
    spec = torch.from_numpy(spec.T)
    if use_cuda:
        spec = spec.cuda()
    spec = spec.unsqueeze(0)
    embed = speaker_encoder.compute_embedding(spec).detach().cpu().numpy()
    embed = embed.squeeze()
    # embed_path = wav_file.replace(dataset_path, output_path)
    # embed_path = embed_path.replace(".wav", ".spk")
    np.save(output_path, embed, allow_pickle=False)