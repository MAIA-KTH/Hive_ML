Hive-ML config
========================================================


.. jsonschema:: configs/Hive_ML_config_template.json
.. code-block:: json

    {
      "image_suffix": "_image.nii.gz",
      "mask_suffix": "_mask.nii.gz",
      "label_dict": {
        "0": "non-pCR",
        "1": "pCR"
      },
      "models": {
        "rf": {
          "criterion": "gini",
          "n_estimators": 100,
          "max_depth": 10
        },
        "adab": {
          "criterion": "gini",
          "n_estimators": 100,
          "max_depth": 10
        },
        "knn": {},
        "lda": {},
        "qda": {},
        "logistic_regression": {},
        "svm": {
          "kernel": "rbf"
        },
        "naive": {}
      },
      "feature_selection": "SFFS",
      "n_features": 30,
      "n_folds": 5,
      "random_seed": 12345,
      "feature_aggregator": "SD"
    }