# Data used for training and testing

Some datasets used for training are very large. If you do not want to retrain the model, you can skip them.

## PF-Pascal (training and testing)

1. Follow the instructions [here](https://github.com/Junyi42/GeoAware-SC/blob/master/data/prepare_pfpascal.sh) to
   download and prepare the dataset. The structure should be:

```
PF-dataset-PASCAL
├──Annotations
├──JPEGImages
├── train_pairs_pf_pascal.csv
├── val_pairs_pf_pascal.csv
└── test_pairs_pf_pascal.csv
```

## PF-Willow (testing only)

1. Download PF-Willow dataset from the [link](https://www.di.ens.fr/willow/research/proposalflow/).
2. Rename the outermost directory from `PF-dataset` to `pf-willow`.
3. Download lists for image pairs from [link](https://www.robots.ox.ac.uk/~xinghui/sd4match/test_pairs.csv).
4. Place the lists for image pairs under `pf-willow` directory. The structure should be:

```
pf-willow
├── PF-dataset
└── test_pairs.csv
```

## SPair-71K (training and testing)

1. Download SPair-71K dataset from [link](https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz). After
   extraction, No more action required. The structure should be:

```
SPair-71k
├── ImageAnnotation
├── JPEGImages
└── PairAnnotation
└──...
```

## AP-10k (training only)

1. Download the AP-10k dataset and extract it.

```
./data/prepare_ap10k.sh
```

2. Follow the instructions provided by GeoAware-SC
   at [here](https://github.com/Junyi42/GeoAware-SC/blob/master/prepare_ap10k.ipynb) to process AP-10k
   dataset. After processing, the structure should be:

```
ap-10k
├── annotations
├──── ap10k-train-split1.json
├──── ap10k-train-split2.json
├──── ap10k-train-split3.json
├──── ...
├── ImageAnnotation
├── JPEGImages
├── PairAnnotation
├──── trn 
├──── test
├──── val
```

## Aachen (testing only)

1. Download Aachen images dataset
   from [here](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/images/database_and_query_images.zip)
   and pairs split from [here](). The structure should be:

```
Aachen
├── aachen_test_1500_pairs.txt
├── database_intrinsics.txt
└── db_poses.txt
└── images_upright
```

## Megadepth (training and testing)

1. Download Megadepth dataset following the instructions from
   LoFTER [here](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md#download-datasets). If you want to do
   testing only, skip this step.
2. Download the test scene info from [here](https://drive.google.com/drive/folders/1nTkK1485FuwqA0DbZrK2Cl0WnXadUZdc).
3. After extraction, the structure should be:

```
megadepth
├── train-data (for training)
└────phoenix
└────megadepth_indices
└──────scene_info_0.1_0.7
└────────0000_0.1_0.3.npz
└────────...
├── megadepth_test_1500 (for testing)
└──── 0015_0.1_0.3.npz
└──── 0015_0.3_0.5.npz
└──── 0022_0.1_0.3.npz
└──── 0022_0.5_0.7.npz
└──── Undistorted_SfM
└──── megadepth_test_1500.txt
```

## Scannet (training and testing)

1. Download scannet dataset from [here](http://www.scan-net.org/). If you want to do testing only, skip this step.
2. Download the test scene info from [here](https://drive.google.com/drive/folders/1nTkK1485FuwqA0DbZrK2Cl0WnXadUZdc).
3. After extraction, the structure should be:

```
scannet
├── scannet_test_1500 (testing)
└────intrinsics.npz
└────scannet_test.txt
└────statistics.json
└────test.npz
```

## TAPVID_DAVIS (testing only)

1. Download TAPVID dataset from [here](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip). After extraction, no
   more action required. The structure should be:

```
tapvid_davis
├── tapvid_davis.pkl
├── README.md
└── SOURCES.md
```

## COCO-20k (training only)

1. Download the COCO-20k subset provided by XFeat
   from [here](https://drive.google.com/file/d/1ijYsPq7dtLQSl-oEsUOGH1fAy21YLc7H/view). After extraction, no more action
   required. The structure should be:

```
coco_20k
├── 000000140092.jpg
└── ...jpg
```
