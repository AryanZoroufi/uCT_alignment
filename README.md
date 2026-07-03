## Installation


1. **Clone the repository**:
    ```sh
    git clone https://github.com/AryanZoroufi/uCT_alignment.git
    cd uCT_alignment
    ```

2. **Create a conda environment**:
    ```sh
    conda create -n uct python=3.10
    conda activate uct
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

    cd uct
    python pipeline.py <path_to_ref_vox> <path_to_sample_vox> b--bones 1 4 -o <save_dir> --volume-scale 1.25e-4 --visualize --neighborhood-radius 0.10 --patch-radius 0.3 --visualize