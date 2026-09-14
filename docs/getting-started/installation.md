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

The package offers a CUDA 12 extra:

```bash
python -m pip install "poscidyn[gpu]"
```

GPU support ultimately depends on the JAX backend, driver, CUDA, and platform
combination. Follow the current [JAX installation guidance](https://docs.jax.dev/en/latest/installation.html)
when configuring a non-CPU backend.

## Verify the installation

```bash
python -c "import poscidyn; print('Poscidyn imported successfully')"
```

Then run the [first frequency sweep](../quickstart/frequency-sweep.md). The
first numerical call may take longer because JAX compiles the computation; later
calls with compatible shapes can reuse that compiled work.
