# ImageRestoration

Old photo restoration based on Microsoft's
[Bringing Old Photos Back to Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)
(MIT License). This copy keeps the **global restoration** stage only — scratch
detection plus quality/scratch restoration. The face-detection and
face-enhancement stages of the original project are not included.

## Pipeline

```
input image ──► detection.py ──► scratch mask ──┐
                (UNet, checkpoints/detection)   ├──► test.py ──► restored image
input image ────────────────────────────────────┘   (VAE + feature mapping,
                                                     checkpoints/restoration)
```

`run.py` is the driver that chains both stages. Without `--with_scratch` it
skips detection and runs quality restoration only.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

For GPU use, install a CUDA build of PyTorch first (see the comment at the top
of `requirements.txt`), then install the rest.

## Pretrained checkpoints

Model weights are **not** in the repository (`*.pt` / `*.pth` are gitignored).
Download `global_checkpoints.zip` from the upstream v1.0 release:

<https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip>

```bash
# Linux / macOS
wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
unzip global_checkpoints.zip
```

```powershell
# Windows PowerShell
Invoke-WebRequest -Uri https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip -OutFile global_checkpoints.zip
Expand-Archive global_checkpoints.zip -DestinationPath .
```

Unpack it in the repository root so the tree looks like this:

```
checkpoints/
├── detection/
│   └── FT_Epoch_latest.pt
└── restoration/
    ├── mapping_Patch_Attention/   # used with --HR
    ├── mapping_quality/
    ├── mapping_scratch/
    ├── VAE_A_quality/
    ├── VAE_B_quality/
    └── VAE_B_scratch/
```

Each `mapping_*` directory needs `latest_net_mapping_net.pth`, each `VAE_*`
directory needs `latest_net_G.pth`.

## Usage

Restore photos without scratches:

```bash
python run.py --input_folder ./test_images --output_folder ./output --GPU 0
```

Restore scratched photos (runs scratch detection first):

```bash
python run.py --input_folder ./test_images --output_folder ./output --GPU 0 --with_scratch
```

High-resolution scratched photos (uses the patch-attention mapping model):

```bash
python run.py --input_folder ./test_images --output_folder ./output --GPU 0 --with_scratch --HR
```

### `run.py` options

| Option | Default | Meaning |
| --- | --- | --- |
| `--input_folder` | `./test_images` | Directory of input images |
| `--output_folder` | `./output` | Where results are written |
| `--GPU` | `0` | GPU id; pass `-1` to run on CPU |
| `--with_scratch` | off | Detect scratches and inpaint them |
| `--HR` | off | Large scratched inputs (requires `--with_scratch`) |

### Output layout

```
output/
├── origin/           # input after resizing, before restoration
├── input_image/      # tensor actually fed to the network
├── restored_image/   # final result
└── masks/            # only with --with_scratch
    ├── input/        # resized input used for detection
    └── mask/         # predicted scratch mask
```

## Running the stages directly

```bash
# scratch detection only
python detection.py --test_path ./test_images --output_dir ./output/masks --input_size full_size --GPU 0

# quality restoration only
python test.py --test_mode Full --Quality_restore --test_input ./test_images --outputs_dir ./output --gpu_ids 0

# scratch + quality restoration, given masks
python test.py --Scratch_and_Quality_restore --test_input ./output/masks/input --test_mask ./output/masks/mask --outputs_dir ./output --gpu_ids 0
```

`--input_size` for detection accepts `full_size` or `scale_256` (the help text
also lists `resize_256`, but `data_transforms` in `detection.py` does not handle it).
`--test_mode` for quality restoration accepts `Full`, `Scale` or `Crop`.

## Project layout

| Path | Contents |
| --- | --- |
| `run.py` | End-to-end driver |
| `detection.py` | Scratch-detection inference |
| `test.py` | Restoration inference |
| `detection_models/` | UNet + synchronized batch norm for detection |
| `detection_util/` | Helpers used by `detection.py` |
| `models/` | pix2pixHD / VAE / feature-mapping networks |
| `options/` | Command-line option definitions |
| `data/` | Dataset loaders (training-time code) |
| `util/` | Shared helpers |
| `checkpoints/` | Pretrained weights (not tracked in git) |
| `test_images/` | Sample inputs |

## Notes

- `util/visualizer.py` is inherited from the upstream training code, is not
  imported by any of the inference scripts, and still imports `scipy.misc` and
  `tensorflow`, which is why neither is in `requirements.txt`.
- Only inference is wired up here; the training entry point of the upstream
  project was not carried over, though `data/` and `options/train_options.py`
  remain.

## License

MIT, following the upstream project. See the per-file Microsoft copyright headers.
