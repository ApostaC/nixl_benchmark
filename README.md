# Simple benchmark for NIXL with different backends

Currently support UCX and Mooncake backends

## Installation

The installation scripts can be found in the `installation/` directory.

### Prerequisites

- uv virtual environment

### Install UCX

```bash
cd installation
bash install-ucx.sh
```

### Install Mooncake

```bash
cd installation
bash install-mooncake.sh
```

### Install NIXL

```bash
cd installation
bash install-nixl.sh
```

## Running the test across different nodes

On node A:

```bash
# Use --backend Mooncake to test Mooncake backend
# Update <Node B's IP> with the actual IP address of node B
python3 benchmark.py --host <Node B's IP> --port 9876 --role creator --operation WRITE --device cpu --backend UCX
```

On node B (could be the same as node A):

```bash
# Use --backend Mooncake to test Mooncake backend
python3 benchmark.py --host 0.0.0.0 --port 9876 --role peer --operation WRITE --device cpu --backend UCX
```


## Latest Results (May 26)

- UCX backend: 46.8 GB/s
- Mooncake backend: Not runnable yet
