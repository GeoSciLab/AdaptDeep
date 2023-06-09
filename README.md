# A Self-supervised Framework for Geophysical Data 
Restoration via Domain Adaptation

This is the official code of [AdaptDeep (A Self-supervised Framework for Geophysical Data)](https://github.com/GeoSciLab/AdaptDeep).

## Datasets
The grid field data for training *sst*, *t2m*, *vo* and *tp* are available from ERA5 hourly data on pressure (https://doi.org/10.24381/cds.bd0915c6) and single levels (https://doi.org/10.24381/cds.adbb2d47) of ERA5. Observational data during the passage of typhoon In-fa for validation is from six meteorological stations in northern Taiwan (https://www.cwb.gov.tw/V8/C/). 

## Code
It should be easy to use main.py for training or testing .

## Environment
  - Python >= 3.6
  - PyTorch, tested on 1.9, but should be fine when >=1.6

## Citation
If you find our code or datasets helpful, please consider citing our related works.
## Contact
If you have questions or suggestions, please open an issue here or send an email to public_wlw@163.com.

## Acknowledgements

This research was funded by the National Natural Science Foundation of China (Grant No. 7075139, U2242201, 42075139, 41305138), the China Postdoctoral Science Foundation (Grant No. 2017M621700), Hunan Province Natural Science Foundation (Grant No. 2021JC0009, 2021JJ30773) and Fengyun Application Pioneering Project (FY-APP-2022.0605).