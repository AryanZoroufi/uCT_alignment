# uCT Alignment: A toolbox for aligning micro_CTs and calculating the injury volume using Probabilistic Programming


## Installation


1. **Clone the repository**:
    ```sh
    git clone https://github.com/AryanZoroufi/uCT_alignment.git
    cd uCT_alignment
    ```

2. **Create a conda environment**:
    ```sh
    conda create -n uct python=3.10
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

    cd uct
    python pipeline.py <path_to_ref_vox> <path_to_sample_vox> bone_1_atlas.stl bone_2_atlas.stl -o <save_dir> --volume-scale 1.25e-4 --articular-percentile 20