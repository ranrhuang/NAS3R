# Datasets

## Training datasets
For training, we mainly use [RealEstate10K](https://google.github.io/realestate10k/index.html) and [DL3DV](https://github.com/DL3DV-10K/Dataset) datasets. We provide the data processing scripts to convert the original datasets to pytorch chunk files which can be directly loaded with this codebase. 

Expected folder structure:

```
├── datasets
│   ├── re10k
│   ├── ├── train
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── ├── test
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── dl3dv
│   ├── ├── train
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── ├── test
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
```

By default, we assume the datasets are placed in `datasets/re10k`, `datasets/dl3dv`, and `datasets/acid`. Otherwise you will need to specify your dataset path with `dataset.DATASET_NAME.roots=[YOUR_DATASET_PATH]` in the running script.

We also provide instructions to convert additional datasets to the desired format.



### RealEstate10K

For experiments on RealEstate10K, we primarily follow [pixelSplat](https://github.com/dcharatan/pixelsplat) and [MVSplat](https://github.com/donydchen/mvsplat) to train and evaluate on 256x256 resolution.

Please refer to [here](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) for acquiring the processed 360p dataset (360x640 resolution).



### DL3DV

In the DL3DV experiments, we trained with RealEstate10k at 256x256 resolution.


For the training set, we use the [DL3DV-480p](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P) dataset (270x480 resolution), where the 140 scenes in the test set are excluded during processing the training set. After downloading the [DL3DV-480p](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P) dataset, You can first use the script [src/scripts/convert_dl3dv_train.py](src/scripts/convert_dl3dv_train.py) to convert the training set, and then run [src/scripts/generate_dl3dv_index.py](src/scripts/generate_dl3dv_index.py) to generate the `index.json` file for the training set.


Please note that you will need to update the dataset paths in the aforementioned processing scripts.
