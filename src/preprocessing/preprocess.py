import pandas as pd
import argparse
import sys,os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/workspace/lucas.ueda/RecursiveSynthVC')
from omegaconf import OmegaConf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from src.preprocessing.preprocess_spec import compute_spec, process_file_spec
from src.preprocessing.preprocess_melspec16 import pred_melspec16
from src.preprocessing.preprocess_crepe import compute_f0
from src.preprocessing.preprocess_sr import resample_wave, process_file
from src.preprocessing.preprocess_flist import print_error
from src.preprocessing.preprocess_whisper import load_model_whisper, pred_whisper

def process_sr_with_thread_pool(wavPath, outPath, sr, thread_num=None):
    files = [f for f in os.listdir(f"{wavPath}") if f.endswith(".wav")]

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = {executor.submit(process_file, wavPath, outPath, sr): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing resampling'):
            future.result()

def process_spec_with_thread_pool(wavPath, outPath, thread_num):
    files = wavPath
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        list(tqdm(executor.map(process_file_spec, files, outPath), total=len(files), desc=f'Processing spec'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--in_df", help="df", dest="in_df", required=True)
    parser.add_argument("-o", "--out", help="out", dest="out", required=True)
    parser.add_argument("-of", "--out_files", help="out files", dest="out_files", required=True)
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)

    args = parser.parse_args()
    
    print(f'Reading df from {args.in_df}')
    print(f'Saving files in {args.out}')
    threads = args.thread_count
    
    try:
        os.makedirs(args.out, exist_ok=True)
        os.makedirs(args.out + '/spec', exist_ok=True)
        os.makedirs(args.out + '/pitch', exist_ok=True)
        os.makedirs(args.out + '/waves-24k', exist_ok=True)
        os.makedirs(args.out + '/waves-16k', exist_ok=True)
        # os.makedirs(args.out + '/melspec16', exist_ok=True)
        os.makedirs(args.out + '/whisper', exist_ok=True)
        print('{spec}, {pitch}, {waves-24k}, {waves-16k} and {whisper} folders were created.')
    except:
        print('It was not possible to create all directories.')
    
    try:
        df = pd.read_csv(args.in_df)
    except FileNotFoundError:
        print(f"File: {args.in_df} not found")
    
    hps = OmegaConf.load("./configs/base.yaml")

    wavs = df.wav_path.values
    speakers = df.speaker.values
    
    print("Starting resampling 24k")
    # process_sr_with_thread_pool(wavs, args.out + '/waves-24k', 24000, thread_num=None)
    for w in tqdm(wavs):
        if w.endswith(".wav"):
            outpath = args.out + '/waves-24k'
            process_file(w, outpath, 24000)
    print("Starting resampling 16k")
    # process_sr_with_thread_pool(wavs, args.out + '/waves-16k', 16000, thread_num=None)
    for w in tqdm(wavs):
        if w.endswith(".wav"):
            outpath = args.out + '/waves-16k'
            process_file(w, outpath, 16000)
            
    wavs_24k = [args.out + '/waves-24k/' + f for f in os.listdir(args.out + '/waves-24k')]
    wavs_16k = [args.out + '/waves-16k/' + f for f in os.listdir(args.out + '/waves-16k')]
    
    print("Starting extracting spectrograms")
    # process_spec_with_thread_pool(wavs_24k, args.out + '/spec', thread_num=None)
    for w in tqdm(wavs_24k):
        if w.endswith(".wav"):
            outpath = args.out + '/spec'
            process_file_spec(hps, w, outpath)

    # OLD: uncoment if u want to recreate melspec16 for on-training usage. Note that: Whisper encodes with pad_or_trim, so it need to be performed here
    # print("Starting extracting melspec16")
    # for w in tqdm(wavs_16k):
    #     if w.endswith(".wav"):
    #         file = w.split('/')[-1].split('.')[0]
    #         out_path = args.out + '/melspec16/' + file + '.m16'
    #         pred_melspec16(w, out_path)
    
    print("Starting extracting whisper")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper = load_model_whisper(os.path.join("./pre_trained_models", "base.pt"), device)
    for w in tqdm(wavs_16k):
        if w.endswith(".wav"):
            file = w.split('/')[-1].split('.')[0]
            out_path = args.out + '/whisper/' + file + '.whs'
            pred_whisper(whisper, w, out_path)
    del whisper
            
    print("Starting extracting torchcrepe")
    for w in tqdm(wavs_16k):
        if w.endswith(".wav"):
            file = w.split('/')[-1].split('.')[0]
            out_path = args.out + '/pitch/' + file + '.pit'
            compute_f0(w, out_path, device)

    print("Saving fileist")

    rootPath = args.out + '/waves-24k'
    print(rootPath)
    all_items = []
    for i, file in enumerate(tqdm(os.listdir(f"./{rootPath}"))):
        if file.endswith(".wav"):
            file = '_'.join(file.split('/')[-3:]).split('.')[0]

            path_wave = f"{args.out}/waves-24k/{file}.wav"
            path_spec = f"{args.out}/spec/{file}.pt"
            path_pitch = f"{args.out}/pitch/{file}.pit.npy"
            # path_melspec16 = f"{args.out}/melspec16/{file}.m16.npy"
            path_whisper = f"{args.out}/whisper/{file}.whs.npy"
            has_error = 0
            if not os.path.isfile(path_wave):
                print_error(path_wave)
                has_error = 1
            if not os.path.isfile(path_spec):
                print_error(path_spec)
                has_error = 1
            if not os.path.isfile(path_pitch):
                print_error(path_pitch)
                has_error = 1
            if not os.path.isfile(path_whisper):
                print_error(path_whisper)
                has_error = 1
            if has_error == 0:
                spk = speakers[i]
                all_items.append(
                    f"{path_wave}|{path_spec}|{path_pitch}|{path_whisper}|{spk}")

    fw = open(f"{args.out_files}", "w", encoding="utf-8")
    for strs in all_items:
        print(strs, file=fw)
    fw.close()
    
    print(f'All files preprocessed. Filelist saved at {args.out_files}')