import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_msssim import ms_ssim

Image.MAX_IMAGE_PIXELS = 500000000

# 读取图片并转换为Tensor，可指定尺寸
def load_image(path, device, resize=None):
    img = Image.open(path).convert('RGB')
    if resize is not None:
        img = img.resize(resize, resample=Image.BICUBIC)
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0).to(device)  # [1, 3, H, W] 并转到GPU

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 先读取第一张图片，用作基准尺寸
img1_pil = Image.open(R"C:\Users\30927\Desktop\CycleGAN-and-pix2pix\checkpoints\mouse_kidney_Trans_HE_512\web\images\epoch199_real_A.png").convert('RGB')
size = img1_pil.size  # (width, height)

# 加载两张图片，并将图片2 resize 成和图片1一样大
img1 = load_image(R"C:\Users\30927\Desktop\CycleGAN-and-pix2pix\checkpoints\mouse_kidney_Trans_HE_512\web\images\epoch199_real_A.png", device)
img2 = load_image(R"C:\Users\30927\Desktop\CycleGAN-and-pix2pix\checkpoints\mouse_kidney_Trans_HE_512\web\images\epoch199_fake_A.png", device, resize=size)

# 计算MS-SSIM
score = ms_ssim(img1, img2, data_range=1.0, size_average=True)
print(f'MS-SSIM相似度: {score.item():.4f}')