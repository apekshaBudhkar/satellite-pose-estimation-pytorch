# Dataset processing

These scripts are used to generate ML training sets from the raw data. Data collection is performed in small segments and saved to disk in individual folders the format `mm_dd_hhmm` (Example: feb_23_1854).
Each data point is identified by a unix timestamp, where the raw input image is saved in the format `<timestamp>.bin` and the corresponding ground truth file is saved as a text file in `<timestamp>.txt`.

# Recommended workflow for training set generation

1. Create plot of the ground truth poses so it's easier to visualize them.
```sh
python3 generate ground_truth_plots.py
```
Manually QA the images and delete ones that have inaccurate poses (either due to issues in the ground truth), or unwanted artefacts such as a person in the frame.

2. Combine processed data from all segments into one large dataset, which will be used for training.
```sh
python3 combine_datasets.py
```

3. Convert into a dataset that is split into train/validation/test for model training and verification.
```sh
python3 split_dataset.py
```



