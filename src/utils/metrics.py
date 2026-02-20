import numpy as np
from typing import List
from skimage.metrics import structural_similarity as ssim_func


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Hitung PSNR antara dua gambar.
    Formula: PSNR = 20 * log10(MAX / sqrt(MSE))
    Return: nilai PSNR dalam dB. Return inf jika MSE = 0.
    Input: numpy uint8 [H,W,C] atau [H,W]
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Hitung SSIM menggunakan skimage.metrics.structural_similarity.
    Input: numpy uint8 [H,W,C]
    Return: nilai SSIM antara 0 dan 1.
    """
    if img1.ndim == 3:
        value = ssim_func(img1, img2, channel_axis=2, data_range=255)
    else:
        value = ssim_func(img1, img2, data_range=255)
    return float(value)


class LPIPSMetric:
    """
    Wrapper untuk LPIPS metric menggunakan library lpips.
    Lazy-load model saat pertama kali digunakan.
    """

    def __init__(self, network: str = "alex"):
        """Inisialisasi dengan nama network (alex atau vgg)."""
        self.network = network
        self._model = None

    def _load_model(self):
        """Lazy-load LPIPS model."""
        try:
            import lpips
            import torch
            self._model = lpips.LPIPS(net=self.network)
            self._model.eval()
        except ImportError:
            raise ImportError("lpips package not installed. Run: pip install lpips")

    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Input: numpy uint8 [H,W,C] BGR
        Konversi ke tensor [-1,1] sebelum hitung LPIPS.
        Return: nilai LPIPS (lebih rendah = lebih mirip).
        """
        import torch

        if self._model is None:
            self._load_model()

        def to_tensor(img: np.ndarray) -> "torch.Tensor":
            img_rgb = img[:, :, ::-1].copy()
            t = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0)
            t = t / 127.5 - 1.0
            return t

        t1 = to_tensor(img1)
        t2 = to_tensor(img2)

        with torch.no_grad():
            score = self._model(t1, t2)
        return float(score.item())


def evaluate_batch(
    sr_images: List[np.ndarray],
    hr_images: List[np.ndarray],
    metrics: List[str] = ["psnr", "ssim", "lpips"],
    lpips_network: str = "alex",
) -> dict:
    """
    Evaluasi batch gambar. Return dict berisi mean, std, median, values per metrik.
    """
    results: dict = {}
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []

    lpips_metric = None
    if "lpips" in metrics:
        lpips_metric = LPIPSMetric(network=lpips_network)

    for sr, hr in zip(sr_images, hr_images):
        if "psnr" in metrics:
            psnr_vals.append(calculate_psnr(sr, hr))
        if "ssim" in metrics:
            ssim_vals.append(calculate_ssim(sr, hr))
        if "lpips" in metrics and lpips_metric is not None:
            lpips_vals.append(lpips_metric.calculate(sr, hr))

    def summarize(vals: List[float]) -> dict:
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "values": vals,
        }

    if "psnr" in metrics and psnr_vals:
        results["psnr"] = summarize(psnr_vals)
    if "ssim" in metrics and ssim_vals:
        results["ssim"] = summarize(ssim_vals)
    if "lpips" in metrics and lpips_vals:
        results["lpips"] = summarize(lpips_vals)

    return results