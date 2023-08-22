# 3 Layer Classification
Implementation of ConvNeXt and Vision Transformer in PyTorch based on paper [A ConvNet for the 2020s] [1] by Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie  
The github form : [Github - Hugging Face] [2]

## Custom Data Prepared
- 訓練集 / 測試集：對應trani/test資料夾，已將原始照片分為9:1。
- 預測集 (Detect)：所有照片全部放在一個資料夾內。

### Hirarchical Classification
- level dict.py：照自己的資料的階層製作。
- target_name：照自己的資料的階層製作。

## Training 
1. 使用X_train_X_X.py 進行訓練。
2. 更改參數:
    - train_dir:訓練資料夾路徑。
    - test_dir:測試資料夾路徑。
    - target_name:欲訓練資料集的類別。

### Train Type
|代號 |python檔 |特徵 |Loss |框架 |
|------|--------|--------|--------|--------|
|A_F_L |A_train_F_L.py |獨立 |Layer Loss |階層式分類(3層的常見分類) |
|B_F_H |B_train_F_H.py |獨立 |Hierachical loss |階層式分類 |
|C_SF_H |C_train_SF_H.py |共享 |Hierachical loss |階層式分類 |

## Detect (多張預測)
1. 使用detect.py 進行照片預測。
2. 更改參數:
    - csv_save_filename：預測結果csv檔的檔名。
    - modelName：訓練好的權重檔路徑。
    - detect_dir：預測照片存放的路徑。

## Data Training Detail
- MyDataSet：dataLoader，直接輸入原大小照片。
- data_transform：自動將輸入圖片resize成224*224。

## HeatMap 熱力圖
- 使用 cam.py 畫圖
    - modelName：更改為已訓練好的權重檔的路徑。
- convnext.py 在 model_s 前加入self.stages = model。
>單張熱力圖
>>- img_path：欲輸入模型的圖片路徑。
>>- CAM_RESULT_PATH：畫好的熱力圖的儲存路徑。
    輸出多張熱力圖
- imgpath：欲輸入模型的資料夾路徑。
- CAM_RESULT_PATH：畫好的熱力圖的儲存資料夾路徑。
- CAM_FALSE_PATH：分類錯誤的熱力圖與原圖儲存路徑。
- CAM_RIGHT_PATH：分類正確的熱力圖與原圖儲存路徑。
- detect集與訓練/測試集的讀取資料夾方式不同。    
### 熱力圖與原圖組合圖
- bg：畫布大小，(原圖width*2,原圖height)，(2,1)排列。
- bg.save：儲存融合原圖與熱力圖的圖片位置。
### 顯示True class的熱力圖
- (line164)取消註解，並註解掉(line165)->熱力圖繪製。
- (line169)取消註解，並註解掉(line170)->熱力圖文字顯示。
### 更改畫熱力圖的模型 
1. model_features:讀取模型最後一層的輸出特徵圖。 
2. fc_weights：獲得fc層的權重。

[1]: https://arxiv.org/abs/2201.03545 "A ConvNet for the 2020s"
[2]: https://github.com/huggingface/pytorch-image-models "Github - Hugging Face"
