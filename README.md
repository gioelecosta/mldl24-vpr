# mldl24-vpr
In this work we address the Visual Place Recognition (VPR) task which consists in localizing a place depicted in a query image. Only computer vision's techniques are exploited. In the first place, we train our model according to the GSV-Cities framework. Next, we test our best model on two benchmark datasets. Finally, we analyze and compare the results obtained from using different model aggregators, loss functions, mining techniques and optimizers. We also visualize some queries and their predictions to better understand the reasoning behind our model's decisions. We show that the model utilizing MixVPR as aggregator outperforms other evaluated configurations.  
Performance obtained training the model on different aggregators:

| Aggregator | R1    | R5    | R10   | R15   | R20   |
|------------|-------|-------|-------|-------|-------|
| AVG        | 53.85 | 69.41 | 75.20 | 78.51 | 80.76 |
| GeM        | 56.66 | 71.66 | 77.56 | 80.73 | 82.73 |
| ConvAP     | 71.64 | 79.06 | 82.03 | 83.92 | 85.14 |
| MixVPR     | **76.27** | **83.50** | **86.60** | **88.43** | **89.42** |

### Results obtained on SF_XS test

| Aggregator | R1    | R5    | R10   | R15   | R20   |
|------------|-------|-------|-------|-------|-------|
| AVG        | 21.30 | 34.30 | 41.30 | 44.70 | 47.20 |
| GeM        | 23.30 | 35.60 | 43.30 | 47.20 | 49.50 |
| ConvAP     | 46.40 | 58.30 | 65.00 | 67.90 | 69.80 |
| MixVPR     | **55.10** | **66.50** | **71.50** | **73.30** | **74.80** |

### Results obtained on Tokyo_XS

| Aggregator | R1    | R5    | R10   | R15   | R20   |
|------------|-------|-------|-------|-------|-------|
| AVG        | 29.21 | 47.62 | 55.24 | 63.17 | 68.25 |
| GeM        | 35.24 | 56.19 | 62.22 | 69.52 | 74.60 |
| ConvAP     | 68.89 | 82.86 | 86.98 | 86.98 | 87.62 |
| MixVPR     | **73.65** | **85.40** | **89.52** | **91.43** | **93.65** |

Further analysis will be shown in our paper.
