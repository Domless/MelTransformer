import itertools
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models import MelTransformer, MelPerceptualLoss, MelDiscriminator, MelTransformer2
from modules.style_encoder import StyleEncoder, MultiResSpecDiscriminator, GeneratorLoss, DiscriminatorLoss
from vocoders.hifigan import Generator as HIFIGAN, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss

from dataset import AudioDataset
from utils.pitch import f0_to_coarse
from utils.utils import (
    plot_spectrograms,
    scan_checkpoint,
    load_checkpoint, 
    prepare_and_save_checkpoints, 
    plot_spectrograms__, 
    get_config, 
    save_checkpoint
)
from utils.mel import MelTranform

torch.set_float32_matmul_precision('high')
device = "cuda" if torch.cuda.is_available() else "cpu"


def full_save(
        checkpoint_path, 
        steps, 
        epoch, 
        generator, 
        optim_g, 
        style_encoder, 
        optim_se, 
        spec_d, 
        optim_spec_d,
    ):
    save_checkpoint(
        "{}/tr_{:08d}".format(checkpoint_path, steps), 
        {
            'generator': generator.state_dict(),
            "optim": optim_g.state_dict(),
            "steps":steps, 
            "epoch": epoch
        }
    )
    save_checkpoint(
        "{}/se_{:08d}".format(checkpoint_path, steps), {
            "encoder": style_encoder.state_dict(),
            "optim": optim_se.state_dict(),
        }
    )
    save_checkpoint(
        "{}/de_{:08d}".format(checkpoint_path, steps), {
            "discriminator": spec_d.state_dict(),
            "optim": optim_spec_d.state_dict(),
        }
    )

# 🔹 Функция обучения
def train_vocoder(dataloader, h, checkpoint_path, tr_h, tr_checkpoint_path, epochs=30, checkpoint_interval=7, new_learning_rate=None):

    mel_generator = MelTransformer2(
        hidden_dim=256, num_layers=4, nhead=16, ideal_dim=256, is_mel_ideal=False
    ).to(device)

    style_encoder = StyleEncoder(dim_in=tr_h.dim_in, style_dim=tr_h.style_dim, max_conv_dim=tr_h.hidden_dim).to(device)
    pitch_embed = torch.nn.Embedding(300, 256, padding_idx=0).to(device)

    cp_g, cp_se = None, None

    if os.path.isdir(tr_checkpoint_path):
        cp_g = scan_checkpoint(tr_checkpoint_path, 'tr_')
        cp_se = scan_checkpoint(tr_checkpoint_path, 'se_')

    if cp_g and cp_se:
        state_dict_g = load_checkpoint(cp_g, device)
        mel_generator.load_state_dict(state_dict_g['generator'])
        state_dict_se = load_checkpoint(cp_se, device)
        style_encoder.load_state_dict(state_dict_se["encoder"])

    mel_generator.eval()
    style_encoder.eval()

    #HIFI GAN
    generator = HIFIGAN(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    cp_g, cp_se = None, None
    if os.path.isdir(checkpoint_path):
        cp_g = scan_checkpoint(checkpoint_path, 'g_')
        cp_do = scan_checkpoint(checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    generator.train()
    mpd.train()
    msd.train()

    mel_tranform = MelTranform.from_h(tr_h, device)
    for epoch in range(max(0, last_epoch), epochs):
        epoch_loss_only = 0.0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} - Train") as pbar:
            for idx, batch in enumerate(pbar, 1):
                x_mel, y_mel, ideal, note, y = batch
                x_mel, y_mel, ideal, note, y= x_mel.to(device).permute(0, 2, 1), y_mel.to(device), ideal.to(device), note.to(device), y.to(device)
                
                ideal_to = style_encoder(ideal.detach())
                pitch_emb = pitch_embed(f0_to_coarse(note))
                dec_inp = torch.stack([pitch_emb, ideal_to.detach().squeeze(1)], 1)
                x = mel_generator(x_mel, dec_inp).permute(0, 2, 1)
                
                y = y.unsqueeze(1)

                y_g_hat = generator(x)
                y_g_hat_mel = mel_tranform.prepare(y_g_hat.squeeze(1))
                y = y[:, :, :y_g_hat.shape[2]]
                y_mel = y_mel[:, :, :y_g_hat_mel.shape[2]]

                optim_d.zero_grad()

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_f

                loss_disc_all.backward()
                optim_d.step()

                # Generator
                optim_g.zero_grad()

                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

                loss_gen_all.backward()
                optim_g.step()

                epoch_loss_only += loss_gen_all.item()
                pbar.set_postfix(
                    loss=loss_gen_all.item(), 
                    d_mel_l1=epoch_loss_only/idx
                )
        
        steps+=1
        scheduler_g.step()
        scheduler_d.step()
        print("Learning rate:", scheduler_g.get_last_lr())

        if steps % checkpoint_interval == 0 and steps != 0:
            path = "{}/g_{:08d}".format(checkpoint_path, steps)
            save_checkpoint(path, {'generator': generator.state_dict()})


        print(f"🔹Step: {steps}, Эпоха: [{epoch+1}/{epochs}], loss: {epoch_loss_only / len(dataloader):.7f}")
    print("✅ Обучение завершено! Сохраняем модель...")
    path = "{}/g_{:08d}".format(checkpoint_path, steps)
    save_checkpoint(path, {'generator': generator.state_dict()})
    # path = "{}/do_{:08d}".format(checkpoint_path, steps)
    # save_checkpoint(path, {
    #             'mpd': mpd.state_dict(),
    #             'msd': msd.state_dict(),
    #             'optim_g': optim_g.state_dict(), 
    #             'optim_d': optim_d.state_dict(), 
    #             'steps': steps,
    #             'epoch': epoch
    #             }
    #         )


def set_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # 🔹 Загружаем данные и запускаем обучение
    h = get_config("./configs/v1.json")
    dataset = AudioDataset(
        "./../prepare/datasets/set_18.05.25", 
        "./../prepare/data/ideals_",
        device, 
        h,
        use_cache=True,
        with_wav=True,
        cahce_folder="./cache_hifi"
    )
    hifi_h = get_config("./configs/hifigan/v1.json")
    set_seed(42)
    #dataset = AudioDataset("./../prepare/datasets/test_set", "./../prepare/data/ideals_", device, h)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)#, num_workers=2, pin_memory=True)
    train_vocoder(dataloader, hifi_h, "./checkpoints_hifi", h, "./checkpoints", epochs=94, checkpoint_interval=20, new_learning_rate=0.00004)