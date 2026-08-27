# Container Images

CDTools is available as a container image on GitHub Container Registry (ghcr.io). The image includes PyTorch with CUDA support and automatically uses GPU acceleration when available, or falls back to CPU mode on systems without GPUs (like Macs).

## Available Image

- **Universal image**: `ghcr.io/als-computing/cdtools:latest`
  - ~3GB with CUDA support
  - Auto-detects and uses GPU when available
  - Falls back to CPU on Macs and systems without GPUs

## Quick Start

### Using with Podman

```bash
# Pull the image
podman pull ghcr.io/als-computing/cdtools:latest

# Run in CPU mode (no GPU flags needed)
podman run -it ghcr.io/als-computing/cdtools:latest

# Run with GPU support (Linux with NVIDIA GPU - requires nvidia-container-toolkit)
podman run --device nvidia.com/gpu=all -it ghcr.io/als-computing/cdtools:latest

# Run an example (CPU mode)
podman run -it ghcr.io/als-computing/cdtools:latest \
  python examples/simple_ptycho.py
```

### Using with Docker

```bash
# Pull the image
docker pull ghcr.io/als-computing/cdtools:latest

# Run in CPU mode (no GPU flags needed - works on Mac too)
docker run -it ghcr.io/als-computing/cdtools:latest

# Run with GPU support (Linux with NVIDIA GPU - requires NVIDIA Container Toolkit)
docker run --gpus all -it ghcr.io/als-computing/cdtools:latest

# Run an example (CPU mode)
docker run -it ghcr.io/als-computing/cdtools:latest \
  python examples/simple_ptycho.py
```

## Mounting Data

To work with your own data files:

```bash
# With Podman (Linux with GPU)
podman run -v /path/to/your/data:/data:z \
  --device nvidia.com/gpu=all -it \
  ghcr.io/als-computing/cdtools:latest \
  python -c "from cdtools.datasets import Ptycho2DDataset; dataset = Ptycho2DDataset.from_cxi('/data/your_file.cxi')"

# With Docker (Linux with GPU)
docker run -v /path/to/your/data:/data \
  --gpus all -it \
  ghcr.io/als-computing/cdtools:latest \
  python -c "from cdtools.datasets import Ptycho2DDataset; dataset = Ptycho2DDataset.from_cxi('/data/your_file.cxi')"

# Without GPU (Podman or Docker)
podman run -v /path/to/your/data:/data:z -it ghcr.io/als-computing/cdtools:latest
docker run -v /path/to/your/data:/data -it ghcr.io/als-computing/cdtools:latest
```

## Interactive Development

```bash
# Start an interactive Python session
podman run -it ghcr.io/als-computing/cdtools:latest

# Start with a bash shell
podman run -it --entrypoint bash ghcr.io/als-computing/cdtools:latest
```

## Version Tags

Images are tagged with:
- `latest` - Latest stable release
- `v0.3.1` - Specific version
- `master` - Latest from master branch

## Platform Support

The image works on:
- ✅ **Linux with NVIDIA GPU** - Full CUDA acceleration
- ✅ **Linux without GPU** - CPU mode (automatic fallback)
- ✅ **macOS (Intel/Apple Silicon)** - CPU mode (automatic fallback)
- ✅ **Windows with WSL2** - GPU or CPU mode

## Image Size

- **Universal image**: ~3.0 GB (includes CUDA libraries for GPU support)

## Building Locally

To build the image locally:

```bash
# Using Podman
podman build -t cdtools:local .

# Using Docker
docker build -t cdtools:local .
```

## Requirements

- **For GPU support**: NVIDIA GPU with CUDA support and nvidia-container-toolkit
- **For CPU mode**: No special requirements (works on any system)
- **Mac users**: The image works out of the box in CPU mode
