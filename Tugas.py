import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import time

def tugas_evaluasi_restorasi():
    print("=" * 60)
    print("EVALUASI RESTORASI CITRA: BLUR & NOISE")
    print("=" * 60)

    # 1. PERSIAPAN CITRA ASLI
    def create_test_image():
        img = np.zeros((256, 256), dtype=np.uint8)
        # Pola grid (frekuensi tinggi)
        for i in range(0, 256, 20):
            cv2.line(img, (i, 0), (i, 255), 100, 1)
            cv2.line(img, (0, i), (255, i), 100, 1)
        # Objek solid
        cv2.rectangle(img, (50, 50), (100, 100), 200, -1)
        cv2.circle(img, (180, 100), 30, 180, -1)
        # Teks
        cv2.putText(img, 'UJI', (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
        return img

    original = create_test_image()

    # 2. GENERASI PSF & DEGRADASI
    def create_motion_blur_psf(length=15, angle=30):
        psf = np.zeros((length, length))
        center = length // 2
        angle_rad = np.deg2rad(angle)
        x_start = int(center - (length/2) * np.cos(angle_rad))
        y_start = int(center - (length/2) * np.sin(angle_rad))
        x_end = int(center + (length/2) * np.cos(angle_rad))
        y_end = int(center + (length/2) * np.sin(angle_rad))
        cv2.line(psf, (x_start, y_start), (x_end, y_end), 1, 1)
        return psf / np.sum(psf)

    psf_true = create_motion_blur_psf(15, 30)
    
    # Skenario 1: Motion Blur Saja
    blurred_pure = cv2.filter2D(original.astype(float), -1, psf_true)
    blurred_pure = np.clip(blurred_pure, 0, 255).astype(np.uint8)

    # Skenario 2: Motion Blur + Gaussian Noise (sigma=20)
    noise_gauss = np.random.normal(0, 20, original.shape)
    blurred_gauss = np.clip(blurred_pure.astype(float) + noise_gauss, 0, 255).astype(np.uint8)

    # Skenario 3: Motion Blur + Salt & Pepper Noise (5%)
    blurred_sp = blurred_pure.copy()
    prob = 0.05
    total_pixels = blurred_sp.size
    num_salt = int(total_pixels * prob / 2)
    salt_coords = [np.random.randint(0, i, num_salt) for i in blurred_sp.shape]
    blurred_sp[salt_coords[0], salt_coords[1]] = 255
    num_pepper = int(total_pixels * prob / 2)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in blurred_sp.shape]
    blurred_sp[pepper_coords[0], pepper_coords[1]] = 0

    scenarios = {
        "S1: Pure Motion Blur": blurred_pure,
        "S2: Blur + Gaussian Noise": blurred_gauss,
        "S3: Blur + S&P Noise": blurred_sp
    }

    # 3. METODE RESTORASI
    def inverse_filter(image, psf, epsilon=1e-3):
        pad_size = psf.shape[0] // 2
        padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        G = np.fft.fft2(padded.astype(float))
        psf_padded = np.zeros_like(padded, dtype=float)
        y_start = padded.shape[0]//2 - psf.shape[0]//2
        x_start = padded.shape[1]//2 - psf.shape[1]//2
        psf_padded[y_start:y_start+psf.shape[0], x_start:x_start+psf.shape[1]] = psf
        psf_padded = np.fft.ifftshift(psf_padded)
        H = np.fft.fft2(psf_padded)
        H_reg = H + epsilon 
        F_hat = G / H_reg
        restored = np.abs(np.fft.ifft2(F_hat))[pad_size:-pad_size, pad_size:-pad_size]
        return np.clip(restored, 0, 255).astype(np.uint8)

    def wiener_filter(image, psf, K=0.01):
        pad_size = psf.shape[0] // 2
        padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        G = np.fft.fft2(padded.astype(float))
        psf_padded = np.zeros_like(padded, dtype=float)
        y_start = padded.shape[0]//2 - psf.shape[0]//2
        x_start = padded.shape[1]//2 - psf.shape[1]//2
        psf_padded[y_start:y_start+psf.shape[0], x_start:x_start+psf.shape[1]] = psf
        psf_padded = np.fft.ifftshift(psf_padded)
        H = np.fft.fft2(psf_padded)
        H_conj = np.conj(H)
        H_abs_sq = np.abs(H) ** 2
        W = H_conj / (H_abs_sq + K)
        F_hat = G * W
        restored = np.abs(np.fft.ifft2(F_hat))[pad_size:-pad_size, pad_size:-pad_size]
        return np.clip(restored, 0, 255).astype(np.uint8)

    def richardson_lucy(image, psf, iterations=20):
        img_float = image.astype(np.float32)
        psf_float = psf.astype(np.float32)
        estimate = img_float.copy()
        psf_flipped = np.flip(psf_float)
        for _ in range(iterations):
            conv = cv2.filter2D(estimate, -1, psf_float)
            conv = np.where(conv == 0, 1e-8, conv)
            ratio = img_float / conv
            correction = cv2.filter2D(ratio, -1, psf_flipped)
            estimate *= correction
            estimate = np.clip(estimate, 0, 255)
        return estimate.astype(np.uint8)

    # 4. METRIK EVALUASI
    def calculate_ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
        mu1, mu2 = cv2.GaussianBlur(img1, (11, 11), 1.5), cv2.GaussianBlur(img2, (11, 11), 1.5)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
        sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1*img2, (11, 11), 1.5) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)

    # 5. EKSEKUSI & EVALUASI
    results = {}
    print(f"{'Skenario':<26} | {'Metode':<15} | {'PSNR (dB)':<10} | {'MSE':<10} | {'SSIM':<8} | {'Waktu (s)':<10}")
    print("-" * 88)

    for scenario_name, deg_img in scenarios.items():
        methods = {
            "Inverse": lambda img: inverse_filter(img, psf_true, epsilon=1e-3),
            "Wiener": lambda img: wiener_filter(img, psf_true, K=0.01),
            "Lucy-Richardson": lambda img: richardson_lucy(img, psf_true, iterations=20)
        }
        
        scenario_results = {}
        for method_name, method_func in methods.items():
            start_time = time.time()
            restored_img = method_func(deg_img)
            calc_time = time.time() - start_time
            
            mse = np.mean((original.astype(float) - restored_img.astype(float)) ** 2)
            psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
            ssim = calculate_ssim(original, restored_img)
            
            scenario_results[method_name] = restored_img
            print(f"{scenario_name:<26} | {method_name:<15} | {psnr:<10.2f} | {mse:<10.2f} | {ssim:<8.3f} | {calc_time:<10.4f}")
            
        results[scenario_name] = scenario_results

    # 6. VISUALISASI HASIL
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for i, (scenario_name, deg_img) in enumerate(scenarios.items()):
        # Kolom 1: Degraded
        axes[i, 0].imshow(deg_img, cmap='gray')
        axes[i, 0].set_title(f"Degraded:\n{scenario_name}")
        axes[i, 0].axis('off')
        
        # Kolom 2-4: Restored
        for j, method_name in enumerate(["Inverse", "Wiener", "Lucy-Richardson"]):
            restored_img = results[scenario_name][method_name]
            axes[i, j+1].imshow(restored_img, cmap='gray')
            axes[i, j+1].set_title(f"Restored: {method_name}")
            axes[i, j+1].axis('off')
            
    plt.tight_layout()
    plt.show()

tugas_evaluasi_restorasi()