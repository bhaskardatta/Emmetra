import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import exposure, restoration
import os
import logging
import rawpy
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.ndimage import gaussian_filter
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise

class EnhancedImageProcessor:
    def __init__(self):
        self.supported_formats = ['.raw', '.RAW', '.dng', '.DNG']
        self.default_params = {'width': 1920, 'height': 1280, 'bits': 14}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dncnn_model = self.initialize_dncnn()

    def initialize_dncnn(self):
        model = DnCNN().to(self.device)
        model.eval()
        return model

    def read_raw_manual(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                data = np.fromfile(f, dtype=np.uint16)
            width = self.default_params['width']
            height = self.default_params['height']
            required_size = width * height
            if len(data) < required_size:
                raise ValueError("File size is smaller than expected dimensions")
            image = data[:required_size].reshape((height, width))
            image = image.astype(np.float32) / (2**self.default_params['bits'] - 1)
            rgb = np.stack([image] * 3, axis=-1)
            return rgb
        except Exception as e:
            logging.error(f"Error reading RAW file manually: {str(e)}")
            raise ValueError(f"Could not read file {file_path}: {str(e)}")

    def read_raw_file(self, file_path):
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, output_bps=16, no_auto_bright=False,
                    use_camera_wb=True, bright=2.0, exp_shift=2.0, gamma=(2.2, 4.5), user_wb=[1.8, 1.0, 1.8, 1],
                    chromatic_aberration=(1.0, 1.0))
                return rgb.astype(np.float32) / 65535.0
        except Exception as e:
            return self.read_raw_manual(file_path)

    def deep_learning_denoise(self, image):
        try:
            transform = transforms.ToTensor()
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                denoised = self.dncnn_model(image_tensor)
            denoised = denoised.squeeze(0).cpu().numpy()
            denoised = np.transpose(denoised, (1, 2, 0))
            denoised = np.clip(denoised, 0, 1)
            return denoised
        except Exception as e:
            logging.error(f"Error in deep learning denoising: {str(e)}")
            return image

    def preprocess_image(self, image):
        image = exposure.adjust_gamma(image, 1.2)
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_l = clahe.apply(l_channel)
        lab[:,:,0] = enhanced_l
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        enhanced_float = enhanced.astype(np.float32) / 255.0
        enhanced_float = np.clip((enhanced_float - 0.1) * 1.5, 0, 1)
        enhanced_float = self.enhance_colors(enhanced_float)
        return enhanced_float

    def enhance_colors(self, image):
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return enhanced.astype(np.float32) / 255.0

    def adaptive_bilateral_filter(self, image, d=9, sigma_space=75):
        result = np.zeros_like(image)
        for i in range(3):
            channel = image[:,:,i]
            local_std = self.estimate_local_noise(channel)
            sigma_color = float(np.mean(local_std) * 2.5)
            sigma_color = max(10, min(sigma_color, 150))
            channel_uint8 = (channel * 255).astype(np.uint8)
            filtered = cv2.bilateralFilter(channel_uint8, d, sigma_color, sigma_space)
            result[:,:,i] = filtered.astype(np.float32) / 255.0
        return result

    def estimate_local_noise(self, channel, window_size=7):
        return gaussian_filter((channel - gaussian_filter(channel, 1.5))**2, window_size)**0.5

    def advanced_denoising(self, image):
        result = np.zeros_like(image)
        for i in range(3):
            channel = (image[:,:,i] * 255).astype(np.uint8)
            denoised = cv2.fastNlMeansDenoising(channel, None, h=10, templateWindowSize=7, searchWindowSize=21)
            result[:,:,i] = denoised / 255.0
        return result

    def enhanced_laplacian(self, image, alpha=0.3):
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0].astype(np.float32)
        lap1 = cv2.Laplacian(l_channel, cv2.CV_32F, ksize=1)
        lap2 = cv2.Laplacian(l_channel, cv2.CV_32F, ksize=3)
        lap3 = cv2.Laplacian(l_channel, cv2.CV_32F, ksize=5)
        laplacian = (0.6 * lap1 + 0.3 * lap2 + 0.1 * lap3)
        edge_mask = cv2.Canny((l_channel).astype(np.uint8), 50, 150)
        edge_mask = cv2.dilate(edge_mask, np.ones((3,3), np.uint8))
        edge_mask = edge_mask.astype(np.float32) / 255.0
        enhancement = alpha * laplacian * (1 - edge_mask)
        l_channel = np.clip(l_channel + enhancement, 0, 255)
        lab[:,:,0] = l_channel
        enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return enhanced.astype(np.float32) / 255.0

    def local_contrast_enhancement(self, image):
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        p2, p98 = np.percentile(l_channel, (2, 98))
        l_channel = exposure.rescale_intensity(l_channel, in_range=(p2, p98))
        lab[:,:,0] = l_channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced.astype(np.float32) / 255.0

    def process_image(self, image):
        image = self.preprocess_image(image)
        results = {}
        metrics = {}
        results['Original'] = image.copy()
        bilateral = self.adaptive_bilateral_filter(image)
        results['Bilateral'] = bilateral
        metrics['Bilateral'] = self.calculate_metrics(image, bilateral)
        denoised = self.advanced_denoising(image)
        results['Advanced Denoised'] = denoised
        metrics['Advanced Denoised'] = self.calculate_metrics(image, denoised)
        dl_denoised = self.deep_learning_denoise(image)
        results['DL Denoised'] = dl_denoised
        metrics['DL Denoised'] = self.calculate_metrics(image, dl_denoised)
        laplacian = self.enhanced_laplacian(image)
        results['Enhanced Laplacian'] = laplacian
        metrics['Enhanced Laplacian'] = self.calculate_metrics(image, laplacian)
        contrast = self.local_contrast_enhancement(image)
        results['Local Contrast'] = contrast
        metrics['Local Contrast'] = self.calculate_metrics(image, contrast)
        combined = self.local_contrast_enhancement(dl_denoised)
        results['Combined Enhancement'] = combined
        metrics['Combined Enhancement'] = self.calculate_metrics(image, combined)
        return results, metrics

    def calculate_metrics(self, original, processed):
        try:
            psnr_value = psnr(original, processed)
            snr_value = self.calculate_snr(original, processed)
            return {'PSNR': float(f"{psnr_value:.2f}"), 'SNR': float(f"{snr_value:.2f}")}
        except Exception:
            return {'PSNR': 0, 'SNR': 0}

    def calculate_snr(self, original, processed):
        noise = original - processed
        signal_power = np.mean(original ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        return 10 * np.log10(signal_power / noise_power)

def main():
    try:
        processor = EnhancedImageProcessor()
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select RAW Image File",
            filetypes=[("RAW files", "*.raw;*.RAW;*.dng;*.DNG"), ("All files", "*.*")])
        if not file_path:
            return
        input_dir = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(input_dir, f"{file_name}_enhanced_results")
        os.makedirs(output_dir, exist_ok=True)
        image = processor.read_raw_file(file_path)
        results, metrics = processor.process_image(image)
        plt.style.use('dark_background')
        num_images = len(results)
        rows = (num_images + 2) // 3
        plt.figure(figsize=(20, 6*rows))
        for idx, (name, img) in enumerate(results.items(), 1):
            safe_name = "".join([c if c.isalnum() else "_" for c in name])
            output_path = os.path.join(output_dir, f"{safe_name}.png")
            cv2.imwrite(output_path, cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            plt.subplot(rows, 3, idx)
            plt.imshow(img)
            if name in metrics:
                metrics_text = f"PSNR: {metrics[name]['PSNR']}\nSNR: {metrics[name]['SNR']}"
                plt.title(f"{name}\n{metrics_text}", color='white', pad=10)
            else:
                plt.title(name, color='white', pad=10)
            plt.axis('off')
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("Image Processing Metrics:\n")
            f.write("=" * 50 + "\n\n")
            for name, metric in metrics.items():
                f.write(f"{name}:\n")
                f.write(f"  PSNR: {metric['PSNR']:.2f}\n")
                f.write(f"  SNR: {metric['SNR']:.2f}\n")
                f.write("-" * 30 + "\n")
        plt.show()
        messagebox.showinfo("Success", f"Processing complete!\nResults saved in:\n{output_dir}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()