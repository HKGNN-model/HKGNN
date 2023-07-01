# HKGNN
This is the code for our Hyper-relational Knowledge Graph Neural Network.

## Description
We implement our HKGNN approach and three baselines: Flashback, GraphFlashback, and STAN, in this repository. Other baselines mentioned in the paper are directly modified in their original code. The processed datasets of NYC for HKGNN, Flashback, and GraphFlashback are now available. Due to the file size limit, the dataset for STAN is currently not available. We will release our extended Foursquare dataset after the acceptance of the paper.

## Requirements
- numpy==1.20.3
- torch==1.10.0
- torch_geometric==1.7.2
- scipy==1.9.3
- pandas==1.1.4
- tqdm==4.64.0
- geohash==1.0

## Useage
To train our HKGNN model, directly use:
```
python main.py
```
If you want to change the model, you can use：
```
python main.py --model MODEL_NAME
```
For other details, please use `-h`. \
It is noteworthy that it requires a GPU with more than 15G memory to run our HKGNN with batch-size 128. You can reduce the batch-size to run the code, however, the performance might drop as well.

## Problems
If you're having trouble installing the geohash package, there's a chance it may not be successful. In that case, you should locate the install package and rename the directory from `Geohash` to `geohash`. Additionally, modify the `__init__.py` to  `from .geohash import decode_exactly, decode, encode` instead of `from geohash import decode_exactly, decode, encode`.\
For this preview version, you can simply annotate  `import geohash` in `data_processor.py` since the dataset is preprocessed.
