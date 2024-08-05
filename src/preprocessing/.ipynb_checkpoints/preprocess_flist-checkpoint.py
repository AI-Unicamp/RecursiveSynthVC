import os
import random
import argparse


def print_error(info):
    print(f"\033[31m File isn't existed: {info}\033[0m")


if __name__ == "__main__":
    os.makedirs("./files/", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", help="root", dest="root", required=True)

    args = parser.parse_args()
    

    
    rootPath = args.root + '/waves-24k'
    print(rootPath)
    
    all_items = []
    for spks in os.listdir(f"./{rootPath}"):
        if not os.path.isdir(f"./{rootPath}/{spks}"):
            continue
        print(f"./{rootPath}/{spks}")
        for file in os.listdir(f"./{rootPath}/{spks}"):
            if file.endswith(".wav"):
                file = file[:-4]

                path_wave = f"{args.root}/waves-24k/{spks}/{file}.wav"
                path_spec = f"{args.root}/specs/{spks}/{file}.pt"
                path_pitch = f"{args.root}/pitch/{spks}/{file}.pit.npy"
#                 path_hubert = f"./data_svc/hubert/{spks}/{file}.vec.npy"
                path_melspec16 = f"{args.root}/melspec16/{spks}/{file}.m16.npy"
                has_error = 0
#                 if not os.path.isfile(path_spk):
#                     print_error(path_spk)
#                     has_error = 1
                if not os.path.isfile(path_wave):
                    print_error(path_wave)
                    has_error = 1
                if not os.path.isfile(path_spec):
                    print_error(path_spec)
                    has_error = 1
                if not os.path.isfile(path_pitch):
                    print_error(path_pitch)
                    has_error = 1
#                 if not os.path.isfile(path_hubert):
#                     print_error(path_hubert)
#                     has_error = 1
                if not os.path.isfile(path_melspec16):
                    print_error(path_melspec16)
                    has_error = 1
                if has_error == 0:
                    all_items.append(
                        f"{path_wave}|{path_spec}|{path_pitch}|{path_melspec16}")

    random.shuffle(all_items)
    valids = all_items[:10]
    valids.sort()
    trains = all_items[10:]
    # trains.sort()
    fw = open("./files/valid.txt", "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()
    fw = open("./files/train.txt", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()
