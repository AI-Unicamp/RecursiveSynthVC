import tqdm
import torch
import torch.nn.functional as F


# +
def validate(hp, args, generator, discriminator, valloader, stft, writer, step, device):
    generator.eval()
    discriminator.eval()
    torch.backends.cudnn.benchmark = False

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    mel_loss = 0.0
    for idx, (melspec16, ppg_l, pit, spec, spec_l, audio, audio_l, melspec, spkids) in enumerate(loader):
        melspec16 = melspec16.to(device)
#         vec = vec.to(device)
        pit = pit.to(device)
        spkids = spkids.to(device)
        ppg_l = ppg_l.to(device)
        audio = audio.to(device)
        melspec = melspec.to(device)
        
#         print(melspec16.shape)
        if hasattr(generator, 'module'):
            fake_audio = generator.module.infer(melspec16, pit, ppg_l, melspec, spkids)[
                :, :, :audio.size(2)]
        else:
            fake_audio = generator.infer(melspec16, pit,ppg_l, melspec, spkids)[
                :, :, :audio.size(2)]
        
#         print(audio.shape, fake_audio.shape)
        mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
        mel_real = stft.mel_spectrogram(audio.squeeze(1))

        mel_loss += F.l1_loss(mel_fake, mel_real).item()

        if idx < hp.log.num_audio:
            spec_fake = stft.linear_spectrogram(fake_audio.squeeze(1))
            spec_real = stft.linear_spectrogram(audio.squeeze(1))

            audio = audio[0][0].cpu().detach().numpy()
            fake_audio = fake_audio[0][0].cpu().detach().numpy()
            spec_fake = spec_fake[0].cpu().detach().numpy()
            spec_real = spec_real[0].cpu().detach().numpy()
            writer.log_fig_audio(
                audio, fake_audio, spec_fake, spec_real, idx, step)

    mel_loss = mel_loss / len(valloader.dataset)

    writer.log_validation(mel_loss, generator, discriminator, step)

    torch.backends.cudnn.benchmark = True
    
    return mel_loss
