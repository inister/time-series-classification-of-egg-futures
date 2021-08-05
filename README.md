# Time-series-classification-of-egg-futures
This is a repository that compromise the datasets and code in our paper *Decision-making Method for Egg Futures Trading Based on Transfer Learning and Hybrid Deep Time Series Classification Model*.
## Environment settings
- Using command pip install -r requirements.txt to install the required library.
## Datesets
- All Datesets in our paper are saved in **all_dataset** folder.
- Before Training and evaluation, you need to copy the dataset that wants to evaluate to **data** folder, and run the **generate_stock_dataset.py** to generate \*.npy file. 
## Training and evaluation
1. For Training and evaluation, you should run **stock_model.py**.
2. If you do not use transfer learning for training and evaluation, you should set parameters:
    - pre_weight_c = None
    - pre_weight_m = None
    - pre_weight_m_and_c = None
3. If you want to use corn futures for transfer learning, you should set parameter **pre_weight_c** to your preweight file path:
    - pre_weight_c = "./weights/stock_use_c_weights.h5"
4. If you want to use soybean meal futures for transfer learning, you should set parameter **pre_weight_m** to your preweight file path:
    - pre_weight_m = "./weights/stock_use_m_weights.h5"
5. If you want to use soybean meal and corn futures for transfer learning, you should set parameter **pre_weight_m_and_c** to your preweight file path:
    - pre_weight_m_and_c = "./weights/stock_use_m_weights.h5"
6. You also can use parameters **year** and **indice** to specify the weight name.
