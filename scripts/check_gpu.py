"""Check CUDA kernels actually execute, not just that a driver is visible."""
import json
import platform
import sys
from importlib.metadata import version


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment. Check NVIDIA driver, CUDA PyTorch, and Jupyter kernel.')
    layer = torch.nn.Conv2d(3, 4, kernel_size=3).cuda()
    inputs = torch.randn(2, 3, 16, 16, device='cuda', requires_grad=True)
    layer(inputs).square().mean().backward()
    matrix = torch.randn(32, 32, device='cuda')
    (matrix @ matrix).sum().item()
    torch.cuda.synchronize()
    return {'python':sys.version, 'platform':platform.platform(), 'torch':str(torch.__version__),
            'torchvision':version('torchvision'), 'transformers':version('transformers'),
            'numpy':version('numpy'), 'pandas':version('pandas'), 'sklearn':version('scikit-learn'),
            'pillow':version('Pillow'), 'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0),
            'vram_bytes':torch.cuda.get_device_properties(0).total_memory}


if __name__ == '__main__':
    print(json.dumps(check_gpu(), indent=2))
    print('GPU convolution, matrix multiplication, and backward PASSED.')
