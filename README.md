# HAM10K
HAM10K Skin Cancer Image Classification

## Literature Review

1. [Dermatologist-level classification of skin cancer with deep neural networks](https://frankie-yanfeng.github.io/2019/01/29/Paper-Summary-Dermatologist-level-classification-of-skin-cancer/)


2. [Poster - A Benchmark for Automatic Visual
Classification of Clinical Skin Disease Images](http://www.eccv2016.org/files/posters/P-4B-40.pdf)


3. [Paper - A Benchmark for Automatic Visual
Classification of Clinical Skin Disease Images](http://cv.nankai.edu.cn/projects/sd-198/2016-ECCV-A%20Benchmark%20for%20Automatic%20Visual%20Classification%20of%20Clinical%20Skin%20Disease%20Images.pdf)

## Objective
Top 3 accuracy exceeding 70%  as a result returned less than 3 seconds

## Challenge
* Unbalanced dataset
* Small data volume

## Method
* Pretrained model (transfer learning) + Retrined fine tuning on own dataset.

* Data augmentation to reduce dataset imbalance and the small amount of data.

## Model Selection
* Originally, there are many choices, like yolo3, SSD, mask-rcnn, etc. But since this challenge is pure mutiple classification task, it is unnecessary to involve object detection and other technologies, because for example, the bounding box regression is taking significant computation time in calculating IOU, meanwhile, HAM10K dataset does not provide bounding box. So, due to literature review solutions and the data volume size, the classical networks like VGG19, ResNet50, etc are shortlisted as potential seletions.


* Data -> Model complexity estimation candidate:


    1. GoogleNet Inception v3 in literature review 1 with 129,450 clinical images(299 X 299) -> 10 times bigger than HAM10K dataset, so Inception v3 is the upper limit in model seletion.
    
    2. VGG 16 is used in literature review 2 & 3. But based on the below comparison, it is very huge without obvious strength.

    3. Taking MobileNet as the bottom limit in model seltion.
    
    4. In order to double confirm the model choice, I may also use DenseNet121 or NASNetMobile for verification.
    

        ![Keras Model Complexity Comparison](https://i.imgur.com/VLtVZnC.png)

## Nice to have features
1. t-SNE for visulization on last hidden layer feature map.
2. cross-validation.
3. Saliency maps.

## Data Processing
* Seven Classes
* 10,000 dermatoscopic images (600 X 450)

## Dataset

[The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions](https://arxiv.org/abs/1803.10417)

## Categories
* nv
        Melanocytic nevi are benign neoplasms of melanocytes and appear in a myriad of variants, which all are included in our series. The variants may differ significantly from a dermatoscopic point of view.[6705 images]


* mel
        Melanoma is a malignant neoplasm derived from melanocytes that may appear in different variants. If excised in an early stage it can be cured by simple surgical excision. Melanomas can be invasive or non-invasive (in situ). We included all variants of melanoma including melanoma in situ, but did exclude non-pigmented, subungual, ocular or mucosal melanoma.[1113 images]

* bkl
        "Benign keratosis" is a generic class that includes seborrheic ker- atoses ("senile wart"), solar lentigo - which can be regarded a flat variant of seborrheic keratosis - and lichen-planus like keratoses (LPLK), which corresponds to a seborrheic keratosis or a solar lentigo with inflammation and regression [22]. The three subgroups may look different dermatoscop- ically, but we grouped them together because they are similar biologically and often reported under the same generic term histopathologically. From a dermatoscopic view, lichen planus-like keratoses are especially challeng- ing because they can show morphologic features mimicking melanoma [23] and are often biopsied or excised for diagnostic reasons.[1099 images]
        
* bcc
        Basal cell carcinoma is a common variant of epithelial skin cancer that rarely metastasizes but grows destructively if untreated. It appears in different morphologic variants (flat, nodular, pigmented, cystic, etc) [21], which are all included in this set.[514 images]

* akiec
        Actinic Keratoses (Solar Keratoses) and intraepithelial Carcinoma (Bowen’s disease) are common non-invasive, variants of squamous cell car- cinoma that can be treated locally without surgery. Some authors regard them as precursors of squamous cell carcinomas and not as actual carci- nomas. There is, however, agreement that these lesions may progress to invasive squamous cell carcinoma - which is usually not pigmented. Both neoplasms commonly show surface scaling and commonly are devoid of pigment. Actinic keratoses are more common on the face and Bowen’s disease is more common on other body sites. Because both types are in- duced by UV-light the surrounding skin is usually typified by severe sun damaged except in cases of Bowen’s disease that are caused by human papilloma virus infection and not by UV. Pigmented variants exists for Bowen’s disease [19] and for actinic keratoses [20]. Both are included in this set.[327 images]

* vasc
        Vascular skin lesions in the dataset range from cherry angiomas to angiokeratomas [25] and pyogenic granulomas [26]. Hemorrhage is also included in this category.[142 images]

* df
        Dermatofibroma is a benign skin lesion regarded as either a benign proliferation or an inflammatory reaction to minimal trauma. It is brown often showing a central zone of fibrosis dermatoscopically [24].[115 images]
        
    [Total images = 10015]

## Experiment 1 and Result Analysis

### Gist
My first experiment start with MobileNet, the code is in <b>Experiment1_[tf_mobilenet_model].ipynb.</b>

The basic ideas are as below:
1. Finding out the repeated images in given dataset, and carefully splitting the train and test dataset in order to make sure there is no duplicated images are taken into test set.

2. Using data augmentation to recude the impact of data imbalance.

3. Stratified split

4. Removing last 5 layers and adding one new dense layer for seven categories classification.

5. Freezing the all layers except the last 23 ones for retraining.

6. Callback involves checkpoint, reduce_lr, earlyStopping

### Outcome


## Experiment 2 and Result Analysis

### Gist
My second experiment is also with MobileNet, the code is in <b>Experiment2_[tf_mobilenet_model].ipynb (tf_mobilenet_model.h5).</b>. The motivation is to run this model again and observe performence change.

### Outcome
After 43 epochs, the training accuracy is very decent which is around 88%, amd the validation one is around 0.73%.
So it seems the first trail is relatively successful. But there are some issues as below:

1. Due to the respective train and validation accuracy, overfitting is still observed.

2. Since this model is run from checkpoint, very little improement is gained, which may be in plateau.

3.  nv & bcc, nv & df and nv & mel, etc are relatively difficult to be differentiated.

4. Even with the boost of data augmentation, the minorities are still in poor performance, which may be explained by lacking of variety amd model memory.
