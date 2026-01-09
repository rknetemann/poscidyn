You can install Poscidyn directly from PyPI using pip:
```bash
pip install poscidyn
```
Requirements:

- Python 3.8 or newer

## Virtual environment
To avoid dependency conflicts with other Python packages, it is strongly recommended to install Poscidyn inside a virtual environment.

### Platform specific setup
Below are examples for the most common platforms.

It is recommended that you create a virtual environment before installing poscidyn in order to avoid potential conflicts with other packages. 

=== "Windows"

    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    ```

=== "Linux / macOS"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

---

## Compatibility

The table below summarizes the supported platforms for **Poscidyn**.

Compatibility is primarily determined by the availability of the **JAX** backend on each platform. As JAX continues to evolve, support for additional platforms may improve over time.

**Legend**
- `yes` — fully supported and verified
- `yes*` — expected to work but not yet fully verified
- `experimental*` — experimental support; functionality may be limited or unstable
- `no` — not supported
- `n/a` — not applicable

| Platform | Linux, x86_64 | Linux, aarch64 | Mac, aarch64 | Windows, x86_64 | Windows WSL2, x86_64 |
|--------|---------------|----------------|--------------|-----------------|----------------------|
| **CPU** | yes | yes* | yes* | yes* | yes* |
| **NVIDIA GPU** | yes | yes* | n/a | no | experimental* |
| **Google Cloud TPU** | yes* | n/a | n/a | n/a | n/a |
| **AMD GPU** | yes* | no | n/a | no | experimental* |
| **Apple GPU** | n/a | no | experimental* | n/a | n/a |
| **Intel GPU** | experimental* | n/a | n/a | no | no |
