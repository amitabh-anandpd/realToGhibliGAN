import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from model4 import GAN
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gan = GAN()
gan.load_state_dict(torch.load("path/to/model4.pth"))
gan.to(device)
gan.eval()
generator = gan.generator
# Preprocess
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

img = Image.open(r"path/to/image.png").convert("RGB")
original_size = img.size
print(original_size)
reverse_transform = transforms.Resize(original_size[::-1])
input_tensor = transform(img).unsqueeze(0).to(device)  # add batch dim

# Generate
generator.eval()
with torch.no_grad():
    output = generator(input_tensor)

# Save or show result
from torchvision.transforms import ToPILImage

output_resized = F.interpolate(output, size=original_size[::-1], mode='bilinear', align_corners=False)

save_image(output, "predicted_image44.png", normalize=True)
