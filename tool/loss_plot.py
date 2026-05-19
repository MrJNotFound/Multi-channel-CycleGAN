import re
import matplotlib.pyplot as plt
from collections import defaultdict
import os

log_path = r"C:\Users\30927\Desktop\CycleGAN-and-pix2pix\checkpoints\mouse_kidney_BF_HE_re_256_instance\loss_log.txt"
save_path = r"C:\Users\30927\Desktop\CycleGAN-and-pix2pix\checkpoints\mouse_kidney_BF_HE_re_256_instance\loss_curve.png"

def plot_losses():
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        epoch_data = {}

        epoch_pattern = re.compile(r'epoch:\s*(\d+)')
        loss_pattern = re.compile(r'([a-zA-Z_]+):\s*([0-9\.]+)')

        for line in lines:
            epoch_match = epoch_pattern.search(line)
            if epoch_match and 'iters:' in line:
                epoch = int(epoch_match.group(1))
                parts = line.split(')')
                if len(parts) > 1:
                    loss_str = parts[-1]
                    matches = loss_pattern.findall(loss_str)
                    if matches:
                        if epoch not in epoch_data:
                            epoch_data[epoch] = {}
                        # 字典的特性会自动覆盖，最终保留的就是该epoch最后一次迭代的loss
                        for name, val in matches:
                            epoch_data[epoch][name] = float(val)

        epochs = sorted(list(epoch_data.keys()))
        total_g_losses = []
        total_d_losses = []

        for ep in epochs:
            ep_losses = epoch_data[ep]
            # 根据GAN网络设计，分为生成器总Loss(Total_G)和判别器总Loss(Total_D)
            # CycleGAN中 G = G_A + G_B + cycle_A + cycle_B + idt_A + idt_B
            # Pix2Pix中 G = G_GAN + G_L1
            # 判别器 D = D_A + D_B 或 D_fake + D_real
            g_loss = 0.0
            d_loss = 0.0
            for k, v in ep_losses.items():
                if k.startswith('D_') or k == 'D':
                    d_loss += v
                else:
                    g_loss += v
            total_g_losses.append(g_loss)
            total_d_losses.append(d_loss)

        def smooth_curve(points, factor=0.85):
            smoothed_points = []
            for point in points:
                if smoothed_points:
                    previous = smoothed_points[-1]
                    smoothed_points.append(previous * factor + point * (1 - factor))
                else:
                    smoothed_points.append(point)
            return smoothed_points

        smooth_g = smooth_curve(total_g_losses)
        smooth_d = smooth_curve(total_d_losses)

        plt.figure(figsize=(12, 7))
        # 原始曲线（调低透明度与线宽）
        plt.plot(epochs, total_g_losses, label='Total Generator Loss (Original)', marker='o', alpha=0.3, zorder=1, color='tab:blue')
        plt.plot(epochs, total_d_losses, label='Total Discriminator Loss (Original)', marker='s', alpha=0.3, zorder=1, color='tab:orange')

        # 平滑曲线（加粗）
        plt.plot(epochs, smooth_g, label='Total Generator Loss (Smoothed)', alpha=1.0, linewidth=2, zorder=2, color='tab:blue')
        plt.plot(epochs, smooth_d, label='Total Discriminator Loss (Smoothed)', alpha=1.0, linewidth=2, zorder=2, color='tab:orange')

        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Training Losses Curve (End of each Epoch)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Loss curve plot saved to: {save_path}")

    except Exception as e:
        print(f"Error plotting losses: {e}")

if __name__ == "__main__":
    plot_losses()
