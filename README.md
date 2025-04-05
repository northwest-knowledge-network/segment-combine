
# combine.py

Combine base PLY, semantic PLY, and instance PLY into a single LAS/LAZ file.

## Usage

combine.py [-h] [--no-compress] base_ply semantic_ply instance_ply output_las

### Positional Arguments

- **base_ply**: Path to the base point cloud PLY file.
- **semantic_ply**: Path to the semantic segmentation PLY file.
- **instance_ply**: Path to the instance segmentation PLY file.
- **output_las**: Path for the output LAS/LAZ file.

### Options

- `-h`, `--help`: Show this help message and exit.
- `--no-compress`: Save as uncompressed LAS instead of LAZ.

