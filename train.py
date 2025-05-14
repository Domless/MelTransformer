import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models import MelTransformer, MelPerceptualLoss, MelDiscriminator
from modules.style_encoder import StyleEncoder, SpecDiscriminator, GeneratorLoss, DiscriminatorLoss

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
def train_vocoder(h, dataloader, dataset, checkpoint_path, epochs=30):
    
    # generator = MelTransformer2(
    #     hidden_dim=128, num_layers=6, nhead=32, ideal_dim=128, is_mel_ideal=True
    # ).to(device)
    generator = MelTransformer(
        hidden_dim=256, num_layers=6, nhead=32, ideal_dim=256, is_mel_ideal=False
    ).to(device)
    # perceptual_loss_fn = MelPerceptualLoss().to(device)
    # discriminator = MelDiscriminator().to(device)
    style_encoder = StyleEncoder(dim_in=h.dim_in, style_dim=h.style_dim, max_conv_dim=h.hidden_dim).to(device)
    spec_d = SpecDiscriminator().to(device)


    pitch_embed = torch.nn.Embedding(300, 256, padding_idx=0).to(device)

    cp_g, cp_se = None, None

    if os.path.isdir(checkpoint_path):
        cp_g = scan_checkpoint(checkpoint_path, 'tr_')
        cp_se = scan_checkpoint(checkpoint_path, 'se_')
        cp_d = scan_checkpoint(checkpoint_path, 'de_')

    last_epoch = -1
    steps = 0
    if cp_g and cp_se:
        state_dict_g = load_checkpoint(cp_g, device)
        generator.load_state_dict(state_dict_g['generator'])
        steps = state_dict_g.get('steps', steps) 
        last_epoch = state_dict_g.get('epoch', last_epoch)

        state_dict_se = load_checkpoint(cp_se, device)
        style_encoder.load_state_dict(state_dict_se["encoder"])
        print(f"load chekpoint {steps} and {last_epoch}")
    
    if cp_d:
        state_dict_d = load_checkpoint(cp_d, device)
        spec_d.load_state_dict(state_dict_d['discriminator'])
        

    optim_g = torch.optim.AdamW(
        generator.parameters(), 
        h.learning_rate, 
        betas=[h.adam_b1, h.adam_b2]
    )
    optim_se = torch.optim.AdamW(
        style_encoder.parameters(),
        h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_spec_d = torch.optim.AdamW(
        spec_d.parameters(),
        h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    if cp_g and cp_se:
        optim_g.load_state_dict(state_dict_g['optim'])
        optim_se.load_state_dict(state_dict_se['optim'])
    if cp_d:
        optim_spec_d.load_state_dict(state_dict_d['optim'])

    g_spec_loss = GeneratorLoss(spec_d).to(device)
    d_spec_loss = DiscriminatorLoss(spec_d).to(device)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_se = torch.optim.lr_scheduler.ExponentialLR(optim_se, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_de = torch.optim.lr_scheduler.ExponentialLR(optim_spec_d, gamma=h.lr_decay, last_epoch=last_epoch)

    spec_d.train()
    generator.train()
    style_encoder.train()
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(max(0, last_epoch), epochs):
        epoch_loss_g_only = 0.0
        epoch_loss_only = 0.0
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} - Train") as pbar:
            for batch in pbar:
                # x, y, y_mel = batch
                x, y, ideal, note = batch
                x, y = x.permute(0, 2, 1), y.permute(0, 2, 1)
                ideal_to = style_encoder(ideal.detach())
                pitch_emb = pitch_embed(f0_to_coarse(note))
                #x, y, z = x.to(device), y.to(device), z.to(device)
                dec_inp = torch.stack([pitch_emb, ideal_to.detach().squeeze(1)], 1)
                y_g_hat = generator(x.detach(), dec_inp.detach())#[:, :, :y.shape[2]]

                # d_real = discriminator(y)
                # d_fake = discriminator(y_g_hat.detach())

                # optim_d.zero_grad()

                # loss_d = torch.mean((d_real - 1)**2) + torch.mean((d_fake)**2)
                # loss_d.backward()
                # optim_d.step()
            
                # L1 Mel-Spectrogram Loss
                mel_l1 = F.l1_loss(y.detach(), y_g_hat) * 10
                epoch_loss_only += mel_l1.item()
                spec_loss = g_spec_loss(y.detach(), y_g_hat.detach())
                #mel_feat = perceptual_loss_fn(y, y_g_hat)
                loss_total = mel_l1 + spec_loss
                #loss_mel = F.l1_loss(y, y_g_hat) * 45
                optim_g.zero_grad()
                loss_total.backward()
                optim_g.step()

                d_loss = d_spec_loss(y.detach(), y_g_hat.detach())
                spec_d.zero_grad()
                d_loss.backward()
                optim_spec_d.step()

                gen_ideal = style_encoder(y.detach().permute(0, 2, 1))
                style_loss = F.l1_loss(ideal_to, gen_ideal)
                optim_se.zero_grad()
                style_loss.backward()
                optim_se.step()


                epoch_loss_g_only += loss_total.item()
                pbar.set_postfix(loss=loss_total.item(), style_loss=style_loss.item(), dis_loss=d_loss.item())
        
        scheduler_g.step()
        scheduler_se.step()
        scheduler_de.step()
        steps+=1
        


        print(f"🔹Step: {steps}, Эпоха: [{epoch+1}/{epochs}], g_only: {epoch_loss_g_only / len(dataloader):.7f}, loss: {epoch_loss_only / len(dataloader):.7f}")
    print("✅ Обучение завершено! Сохраняем модель...")
    for idx, d in enumerate(y_g_hat):
        plot_spectrograms__(
            [
                x[idx].permute(1, 0).detach().cpu().numpy(), 
                y[idx].permute(1, 0).detach().cpu().numpy(), 
                d.permute(1, 0).detach().cpu().numpy(),
            ], 
            ["x", "y", "res"]
        )
    full_save(checkpoint_path, steps, epoch, generator, optim_g, style_encoder, optim_se, spec_d, optim_spec_d)


if __name__ == "__main__":
    # 🔹 Загружаем данные и запускаем обучение
    h = get_config("./configs/v1.json")
    dataset = AudioDataset(
        "./../prepare/datasets/set_augs", 
        "./../prepare/data/ideals_",
        device, 
        h,
        use_cache=True
    )
    #dataset = AudioDataset("./../prepare/datasets/test_set", "./../prepare/data/ideals_", device, h)
    dataloader = DataLoader(dataset, batch_size=h.batch_size, shuffle=True)
    train_vocoder(h, dataloader, dataset, "./checkpoints", epochs=10)