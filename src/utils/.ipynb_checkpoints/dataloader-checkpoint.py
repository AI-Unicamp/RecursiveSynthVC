from torch.utils.data import DataLoader
from src.data.data_utils import DistributedBucketSampler
from src.data.data_utils import TextAudioSpeakerCollate
from src.data.data_utils import TextAudioSpeakerSet


def create_dataloader_train(hps, n_gpus, rank):
    collate_fn = TextAudioSpeakerCollate()
    train_dataset = TextAudioSpeakerSet(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [150, 300, 450],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler)
    return train_loader


def create_dataloader_eval(hps):
    collate_fn = TextAudioSpeakerCollate()
    eval_dataset = TextAudioSpeakerSet(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=8,
        shuffle=False,
        batch_size=hps.train.eval_batch_size,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn)
    return eval_loader