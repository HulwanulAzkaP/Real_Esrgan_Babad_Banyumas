import cv2
import numpy as np
import random
from typing import Tuple


class DegradationPipeline:
    """
    Pipeline degradasi bertingkat (high-order) untuk simulasi degradasi
    naskah kuno dunia nyata. Mengikuti metodologi Real-ESRGAN:
    first-order -> second-order -> sinusoidal -> final downscale.
    """

    def __init__(self, config: dict):
        """Inisialisasi dari konfigurasi degradasi."""
        self.config = config
        self.deg_cfg = config.get("degradation", {})
        self.dataset_cfg = config.get("dataset", {})

    def apply_gaussian_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian blur. sigma dari config gaussian_sigma_range."""
        sigma = float(sigma)
        ks = max(3, int(6 * sigma + 1))
        if ks % 2 == 0:
            ks += 1
        return cv2.GaussianBlur(image, (ks, ks), sigma)

    def apply_motion_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Motion blur dengan kernel size acak dan sudut acak.
        Buat kernel linear dengan sudut random 0-360 derajat.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0
        kernel = kernel / kernel_size

        angle = random.uniform(0, 360)
        center = (kernel_size // 2, kernel_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()

        return cv2.filter2D(image, -1, kernel)

    def apply_anisotropic_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Gaussian blur anisotropik dengan sigma berbeda di arah x dan y.
        sigma_x dan sigma_y masing-masing dipilih acak dari gaussian_sigma_range.
        """
        sigma_range = self.deg_cfg.get("first_order", {}).get("blur", {}).get("gaussian_sigma_range", [0.1, 3.0])
        sigma_x = float(random.uniform(sigma_range[0], sigma_range[1]))
        sigma_y = float(random.uniform(sigma_range[0], sigma_range[1]))
        ks_x = max(3, int(6 * sigma_x + 1))
        ks_y = max(3, int(6 * sigma_y + 1))
        if ks_x % 2 == 0:
            ks_x += 1
        if ks_y % 2 == 0:
            ks_y += 1
        return cv2.GaussianBlur(image, (ks_x, ks_y), sigmaX=sigma_x, sigmaY=sigma_y)

    def apply_resize_degradation(self, image: np.ndarray) -> np.ndarray:
        """
        Resize ke ukuran kecil lalu kembalikan ke ukuran semula,
        mensimulasikan kehilangan detail akibat resolusi rendah.
        """
        resize_cfg = self.deg_cfg.get("first_order", {}).get("resize", {})
        methods = resize_cfg.get("methods", ["bicubic", "bilinear", "area", "nearest"])
        scale_range = resize_cfg.get("scale_range", [0.15, 1.0])

        method_map = {
            "bicubic": cv2.INTER_CUBIC,
            "bilinear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
        }

        method = random.choice(methods)
        scale = random.uniform(scale_range[0], scale_range[1])
        interp = method_map.get(method, cv2.INTER_CUBIC)

        h, w = image.shape[:2]
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))

        small = cv2.resize(image, (new_w, new_h), interpolation=interp)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        return restored

    def apply_gaussian_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        """Tambahkan Gaussian noise dengan std tertentu. Clip ke [0,255]."""
        std = float(std)
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_poisson_noise(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Tambahkan Poisson noise. Scale mengontrol intensitas noise."""
        scale = float(scale)
        img_float = image.astype(np.float32) / 255.0
        noisy = np.random.poisson(img_float * 255.0 * scale) / (255.0 * scale)
        noisy = np.clip(noisy, 0, 1) * 255.0
        return noisy.astype(np.uint8)

    def apply_jpeg_compression(self, image: np.ndarray, quality: int) -> np.ndarray:
        """
        Simulasi artefak JPEG compression.
        Encode ke JPEG dengan quality tertentu, decode kembali ke numpy.
        """
        quality = int(quality)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded

    def apply_sinusoidal_pattern(self, image: np.ndarray) -> np.ndarray:
        """
        Tambahkan pola interferensi sinusoidal (simulasi moire/banding).
        Buat meshgrid dengan frekuensi acak, tambahkan sebagai noise lemah.
        """
        sin_cfg = self.deg_cfg.get("sinusoidal", {})
        freq_range = sin_cfg.get("frequency_range", [6, 16])
        freq = random.uniform(freq_range[0], freq_range[1])

        h, w = image.shape[:2]
        x = np.linspace(0, 2 * np.pi * freq, w, dtype=np.float32)
        y = np.linspace(0, 2 * np.pi * freq, h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        pattern = (np.sin(xx) + np.sin(yy)) * 5.0

        img_float = image.astype(np.float32) + pattern[:, :, np.newaxis]
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def apply_first_order(self, image: np.ndarray) -> np.ndarray:
        """
        Terapkan first-order degradation secara berurutan:
        1. Blur, 2. Resize, 3. Noise, 4. JPEG compression.
        """
        first_cfg = self.deg_cfg.get("first_order", {})
        blur_cfg = first_cfg.get("blur", {})
        noise_cfg = first_cfg.get("noise", {})
        jpeg_cfg = first_cfg.get("jpeg", {})

        # Blur
        if random.random() < blur_cfg.get("probability", 0.8):
            blur_types = blur_cfg.get("types", ["gaussian"])
            blur_type = random.choice(blur_types)
            if blur_type == "gaussian":
                sigma_range = blur_cfg.get("gaussian_sigma_range", [0.1, 3.0])
                sigma = float(random.uniform(sigma_range[0], sigma_range[1]))
                image = self.apply_gaussian_blur(image, sigma)
            elif blur_type == "motion":
                ks_range = blur_cfg.get("motion_kernel_size_range", [7, 21])
                ks = random.randint(ks_range[0], ks_range[1])
                image = self.apply_motion_blur(image, ks)
            elif blur_type == "anisotropic_gaussian":
                image = self.apply_anisotropic_gaussian_blur(image)

        # Resize degradation
        image = self.apply_resize_degradation(image)

        # Noise
        if random.random() < noise_cfg.get("probability", 0.7):
            noise_types = noise_cfg.get("types", ["gaussian", "poisson"])
            noise_type = random.choice(noise_types)
            if noise_type == "gaussian":
                std_range = noise_cfg.get("gaussian_std_range", [0, 25])
                std = float(random.uniform(std_range[0], std_range[1]))
                image = self.apply_gaussian_noise(image, std)
            elif noise_type == "poisson":
                scale_range = noise_cfg.get("poisson_scale_range", [0.05, 3.0])
                scale = float(random.uniform(scale_range[0], scale_range[1]))
                image = self.apply_poisson_noise(image, scale)

        # JPEG compression
        if random.random() < jpeg_cfg.get("probability", 0.6):
            q_range = jpeg_cfg.get("quality_range", [30, 95])
            quality = random.randint(q_range[0], q_range[1])
            image = self.apply_jpeg_compression(image, quality)

        return image

    def apply_second_order(self, image: np.ndarray) -> np.ndarray:
        """
        Terapkan second-order degradation (variasi lebih ringan dari first-order).
        """
        second_cfg = self.deg_cfg.get("second_order", {})
        first_cfg = self.deg_cfg.get("first_order", {})

        blur_prob = second_cfg.get("blur_probability", 0.5)
        noise_prob = second_cfg.get("noise_probability", 0.4)
        jpeg_prob = second_cfg.get("jpeg_probability", 0.5)

        blur_cfg = first_cfg.get("blur", {})
        noise_cfg = first_cfg.get("noise", {})
        jpeg_cfg = first_cfg.get("jpeg", {})

        if random.random() < blur_prob:
            sigma_range = blur_cfg.get("gaussian_sigma_range", [0.1, 3.0])
            sigma = float(random.uniform(sigma_range[0], sigma_range[1]))
            image = self.apply_gaussian_blur(image, sigma)

        image = self.apply_resize_degradation(image)

        if random.random() < noise_prob:
            std_range = noise_cfg.get("gaussian_std_range", [0, 25])
            std = float(random.uniform(std_range[0], std_range[1]))
            image = self.apply_gaussian_noise(image, std)

        if random.random() < jpeg_prob:
            q_range = jpeg_cfg.get("quality_range", [30, 95])
            quality = random.randint(q_range[0], q_range[1])
            image = self.apply_jpeg_compression(image, quality)

        return image

    def apply_full_pipeline(self, hr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Jalankan pipeline lengkap pada HR image.
        Return: (lr_image, hr_image) sebagai pasangan dataset.
        """
        second_cfg = self.deg_cfg.get("second_order", {})
        sin_cfg = self.deg_cfg.get("sinusoidal", {})
        lr_scale = self.dataset_cfg.get("lr_scale", 4)

        image = hr_image.copy()

        # First-order degradation
        image = self.apply_first_order(image)

        # Second-order degradation
        if second_cfg.get("enabled", True):
            image = self.apply_second_order(image)

        # Sinusoidal pattern
        if sin_cfg.get("enabled", True):
            if random.random() < sin_cfg.get("probability", 0.2):
                image = self.apply_sinusoidal_pattern(image)

        # Final bicubic downscale
        h, w = hr_image.shape[:2]
        lr_h = max(1, h // lr_scale)
        lr_w = max(1, w // lr_scale)
        lr_image = cv2.resize(image, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

        return lr_image, hr_image