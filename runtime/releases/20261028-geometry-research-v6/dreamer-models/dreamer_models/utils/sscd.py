from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import PIL.Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.hub import download_url_to_file


class SSCDEmbedder:
    DEFAULT_MODELS = {
        'sscd_disc_mixup': 'https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt',
        'sscd_disc_large': 'https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt',
    }

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_name: str = 'sscd_disc_mixup',
        cache_dir: Optional[Union[str, Path]] = None,
        resize: int = 288,
        device: str = 'cpu',
        download: bool = True,
    ):
        self.device = torch.device(device)
        self.resize = resize
        self.model_path = self._resolve_model_path(model_path, model_name, cache_dir, download)
        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.pil_transform = transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        if hasattr(self.model, 'parameters'):
            for param in self.model.parameters():
                param.requires_grad_(False)

    def _resolve_model_path(self, model_path, model_name, cache_dir, download):
        if model_path is not None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f'SSCD model does not exist: {path}')
            return path

        if model_name not in self.DEFAULT_MODELS:
            raise ValueError(f'Unsupported SSCD model name: {model_name}')

        cache_dir = Path(cache_dir) if cache_dir is not None else Path.home() / '.cache' / 'sscd'
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_name = f'{model_name}.torchscript.pt'
        path = cache_dir / file_name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'SSCD model is missing: {path}')
            download_url_to_file(self.DEFAULT_MODELS[model_name], str(path))
        return path

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = self.model(batch)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        return F.normalize(outputs, dim=-1)

    def _preprocess_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError(f'Expected [N, 3, H, W], got {tuple(batch.shape)}')
        batch = batch.to(device=self.device, dtype=torch.float32)
        batch = F.interpolate(batch, size=(self.resize, self.resize), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=batch.device)[None, :, None, None]
        return (batch - mean) / std

    def encode_targets(self, images: Sequence[PIL.Image.Image]) -> torch.Tensor:
        if len(images) == 0:
            raise ValueError('SSCD target images must not be empty.')
        batch = torch.stack([self.pil_transform(image.convert('RGB')) for image in images], dim=0).to(self.device)
        with torch.no_grad():
            embeddings = self._forward(batch)
        return F.normalize(embeddings.mean(dim=0), dim=0)

    def encode_pil_video(self, frames: Sequence[PIL.Image.Image]) -> torch.Tensor:
        if len(frames) == 0:
            raise ValueError('SSCD video frames must not be empty.')
        batch = torch.stack([self.pil_transform(frame.convert('RGB')) for frame in frames], dim=0).to(self.device)
        with torch.no_grad():
            embeddings = self._forward(batch)
        return F.normalize(embeddings.mean(dim=0), dim=0)

    def encode_frames_for_grad(self, frames: torch.Tensor) -> torch.Tensor:
        batch = self._preprocess_tensor_batch(frames)
        embeddings = self._forward(batch)
        return F.normalize(embeddings.mean(dim=0), dim=0)

    def score_video(self, frames: torch.Tensor, target_embedding: torch.Tensor) -> torch.Tensor:
        video_embedding = self.encode_frames_for_grad(frames)
        return self.cosine_sim(video_embedding, target_embedding)

    @staticmethod
    def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
