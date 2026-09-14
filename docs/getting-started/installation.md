# Installation

Poscidyn requires Python 3.10 or newer. Create an isolated environment, then
install the CPU package:

=== "Linux / macOS"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install poscidyn
    ```

=== "Windows"

    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install poscidyn
    ```

## GPU installation

The recommended GPU extra follows JAX's CUDA 13 wheel:

```bash
python -m pip install "poscidyn[gpu]"
```

For an existing CUDA 12 environment, install the explicit compatibility extra:

```bash
python -m pip install "poscidyn[cuda12]"
```

GPU support ultimately depends on the JAX backend, driver, CUDA, and platform
combination. Follow the current [JAX installation guidance](https://docs.jax.dev/en/latest/installation.html)
when configuring a non-CPU backend.

## Platform compatibility

Poscidyn inherits JAX's platform support. The [JAX supported-platform table](https://docs.jax.dev/en/latest/installation.html#supported-platforms)
is the authoritative and frequently updated compatibility reference; consult it
before selecting hardware or an accelerator backend.

In short, standard CPU installations are available across Linux, macOS, and
Windows, although JAX describes the native Windows x86_64 wheel as
experimental. Pre-built NVIDIA CUDA wheels are available for Linux; native
Windows is not a supported NVIDIA-GPU target. Apple, AMD, TPU, and Intel GPU
backends each have their own support level and installation path in the JAX
table.

### Windows: use WSL2 for NVIDIA acceleration

For Windows users who need NVIDIA GPU acceleration, **WSL2 is the strongly
recommended route**. Work inside a Linux distribution (for example, Ubuntu)
and follow the Linux installation commands there. This gives Poscidyn access to
the Linux CUDA wheel path rather than the unsupported native-Windows GPU path.

If WSL2 is not set up yet, start with Microsoft's
[Install WSL guide](https://learn.microsoft.com/en-us/windows/wsl/install).
It covers the `wsl --install` route and first-time Linux-distribution setup.

JAX currently labels NVIDIA GPU support in WSL2 as experimental, so validate
your particular driver, GPU, and workload before relying on it for production
calculations. After installing the current NVIDIA Windows driver with WSL
support and a WSL2 Linux distribution, run:

```bash
python -m pip install "poscidyn[gpu]"
python -c "import jax; print(jax.devices())"
```

The second command should list a GPU-backed JAX device. If it lists only a CPU,
return to the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
to verify the driver and backend setup.

## Verify the installation

```bash
python -c "import poscidyn; print('Poscidyn imported successfully')"
```

Then run the [first frequency sweep](../quickstart/frequency-sweep.md). The
first numerical call may take longer because JAX compiles the computation; later
calls with compatible shapes can reuse that compiled work.
