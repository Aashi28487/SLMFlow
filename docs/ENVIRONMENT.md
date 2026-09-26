| Property | Measured value |
|---|---|
| Device tier | t1_laptop |
| CPU | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel |
| Logical cores | 16 |
| Total RAM | 16.87 GB |
| Available RAM | 2.56 GB |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Total VRAM | 6.44 GB |
| CUDA available | yes |
| Free disk | 117.84 GB |
| Platform | Windows 10 (AMD64) |
| Python | 3.10.0 |
| Probed at | 2026-09-12T18:12:30+00:00 |

| Field | Read from |
|---|---|
| cpu_model | `env:PROCESSOR_IDENTIFIER` |
| disk | `shutil.disk_usage(C:\Coding\SLMFlow)` |
| gpu | `nvidia-smi` |
| ram | `psutil.virtual_memory` |
