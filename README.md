# 4D_radiomics
The filenames are in sequential orders:
0) feature extraction from the volumetric images
1) using conventional machine learning algorithms for classification based on the extracted features over the train/val set (cross validation)
2) performing a forward feature selection (ffs) to select a subset of most informative features
3) using only the selected features for classification over the train/val set (cross validation)
4) based on the best params of previous experiments, predicting the class labels of test sets. 
