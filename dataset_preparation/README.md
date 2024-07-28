# Dataset processing

These scripts are used to generate ML training sets from the raw data. Data collection is performed in small segments and saved to disk in individual folders the format `mm_dd_hhmm` (Example: feb_23_1854).
Each data point is identified by a unix timestamp, where the raw input image is saved in the format `<timestamp>.bin` and the corresponding ground truth file is saved as a text file in `<timestamp>.txt`.

# Order

1. python3 generate ground_truth_plots.py
2. Manually QA the images and delete ones that are bad
3. python3 combine_datasets.py
4. python3 split_dataset.py
5. python3 copy_raw_markers.py
