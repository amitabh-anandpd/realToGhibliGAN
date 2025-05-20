import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataloader import GhibliDataset
from model4 import GAN


def perceptual_loss(gen, target):
    gen_vgg_feat = vgg(gen)
    target_vgg_feat = vgg(target)
    return F.l1_loss(gen_vgg_feat, target_vgg_feat)

def train(gan, dataloader, num_epochs, device):
    print("Starting Training")
    gan.generator.to(device)
    gan.discriminator.to(device)
    
    gan.generator.train()
    gan.discriminator.train()

    num_batches = len(dataloader)
    # Optimizers
    g_optimizer = optim.Adam(gan.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(gan.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    g_scheduler = StepLR(g_optimizer, step_size=30, gamma=0.5)
    d_scheduler = StepLR(d_optimizer, step_size=30, gamma=0.5)

    least_g_loss = float('inf')

    for epoch in range(num_epochs):
        total_g_loss = 0
        total_d_loss = 0
        total_g_l1_loss = 0
        total_g_adv_loss = 0
        for i, (input_image, target_image) in enumerate(dataloader):
            input_image = input_image.to(device)
            target_image = target_image.to(device)

            gan.discriminator.zero_grad()

            # Real images
            real_output = gan.discriminator(target_image)
            real_labels = torch.ones_like(real_output) * 0.9
            d_loss_real = bce_loss_fn(real_output, real_labels)

            # Fake images
            fake_image = gan.generator(input_image).detach()
            fake_output = gan.discriminator(fake_image)
            fake_labels = torch.zeros_like(fake_output) + 0.1
            d_loss_fake = bce_loss_fn(fake_output, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            d_optimizer.step()
            
            gan.generator.zero_grad()

            generated_image = gan.generator(input_image)
            fake_output = gan.discriminator(generated_image)
            valid_labels = torch.ones_like(fake_output)

            g_adv_loss = bce_loss_fn(fake_output, valid_labels)
            g_l1_loss = l1_loss_fn(generated_image, target_image)
            g_perc_loss = perceptual_loss(generated_image, target_image)
            g_loss = g_adv_loss + lambda_l1 * g_l1_loss + lambda_perc * g_perc_loss

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_g_l1_loss += g_l1_loss.item()
            total_g_adv_loss += g_adv_loss.item()

            g_loss.backward()
            g_optimizer.step()

        avg_loss = total_g_loss/num_batches
        if  avg_loss < least_g_loss:
            torch.save(gan.state_dict(), 'model4.pth')
            least_g_loss = avg_loss
        # Logging
        print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"D Loss: {total_d_loss/num_batches:.4f}, G Loss: {avg_loss:.4f}, "
                f"L1: {total_g_l1_loss/num_batches:.4f}, Adv: {total_g_adv_loss/num_batches:.4f}")
        g_scheduler.step()
        d_scheduler.step()
    torch.save(gan.state_dict(), 'ghibli_model4.pth')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing to [-1, 1]
])

train_dataset = GhibliDataset(root_dir='/path/to/dataset', mode='training', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

l1_loss_fn = nn.L1Loss()
bce_loss_fn = nn.BCELoss()

lambda_l1 = 10
lambda_perc = 0.1

vgg = torchvision.models.vgg19(pretrained=True).features[:16].to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

gan = GAN(in_channels=3, base_channels=64).to(device)

train(
    gan,
    dataloader=train_loader,
    num_epochs=100,
    device=device
)