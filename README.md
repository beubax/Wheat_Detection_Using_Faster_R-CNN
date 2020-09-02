# Implementing Backbones Not Defined in Pytorch's Faster R-CNN Library

Online coding competitions have always been a fascinating _region of interest_ we have wanted to dive into. Kaggle—being amongst the best sources for machine learning related problems—was the first website we stumbled upon. A certain ‘Wheat Head Detection’ caught our eyes, and we had to take a second look at it. The successful implementation of the YOLOv3 algorithm under our arsenal had left us wanting to tackle other methodologies for Object Detection. This competition was the perfect opportunity to dip our feet into the pool of competitive programming, as well expand our understanding of Computer Vision. So, let’s cut the _crop_ and get straight to the implementation. 	


## Table of Contents


- [Why We Decided To Use Faster R-CNN](#why-we-decided-to-use-faster-r-cnn)
- [Training of Faster R-CNN Using Pytorch](#training-of-faster-r-cnn-using-pytorch)
- [Experimenting With Different Backbones On Faster R-CNN](#experimenting-with-different-backbones-on-faster-r-cnn)
- [Inference With Pseudo Labeling](#inference-with-pseudo-labeling)

---

## Why We Decided To Use Faster R-CNN

The first couple of days was spent researching the different available algorithms for object detection and picking the one best suited for our problem statement. R-CNNs aren’t built for speed but rather excel at accuracy. The competition required accurate detection of wheat heads in images, so speed wasn’t of the essence here.  

The algorithm designed for Faster R-CNN (FRCNN) went through several revisions before it became the model it is today. From the inception of the first model—the R-CNN network—that proposed the idea of region-based localization through the Selective Search algorithm to the development of the Fast R-CNN model that improved the speed and accuracy of the R-CNN network, the Faster R-CNN model is the finely tuned network of its predecessors. The model uses a Region Proposal Network to improve on the Selective Search algorithm among several other modifications.

[This article]( https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/) provides an excellent explanation of the exact working of the algorithm.

The ability to detect tiny objects in images was key to detecting wheat heads in the dataset, so Faster R-CNN was our go-to model.

## Training of Faster R-CNN Using Pytorch

The dataset was provided by the competition owners and consisted of ~3400 images of wheat crops taken in different locations around the world. They were labelled and coordinates of the bounding boxes were provided in a CSV file. We tried and tested many different implementations of FRCNN using Keras and Tensorflow, but all of them were either outdated or throwing incomprehensible errors. Pytorch came to the rescue and our code was largely derived from this notebook [Pytorch Faster-R-CNN with ResNet152 backbone](https://www.kaggle.com/maherdeebcv/pytorch-faster-r-cnn-with-resnet152-backbone)

## Experimenting With Different Backbones On Faster R-CNN

One of the advantages of using a model like FRCNN is the flexibility to use different backbones. FRCNN is said to do a better job in locating small objects compared to models such as YOLOv3 and the ability to modify its backbone opens a whirlwind of possibilities. With this in mind, we set out to experiment with every single backbone that managed to run within Kaggle’s free GPU limits!

As mentioned in this [link](https://www.kaggle.com/c/global-wheat-detection/discussion/150972), to modify the backbone for models that do not store feature extraction as a variable, the following code will have to be used:
```shell
class NewNet(NewNet):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)
    def forward(self, inputs):
        # Modify the forward method, so that it returns only the features.
        return super().extract_features(inputs)
backbone = NewNet.from_pretrained(model_name='newnet', num_classes=2)
backbone.out_channels = 1280 
```
To call the model use the following line of code: 

```shell
model = FasterRCNN(backbone, num_classes=2, ...extra params...)
```

To fill the classes with the required code, you will have to go through the code for each backbone. However, the whole process confused us for quite a bit before we got fully functional models up and running. To make it easier to understand, here’s how we worked on the code for SqueezeNet.

- Go to squeezenet.py that can be found in vision/torchvision/models/ on GitHub
- Look for the class SqueezeNet()

![](https://github.com/The-DL-Nerds/Global_Weed_Detection/blob/master/README_Img/2.PNG)

- From the function, look for the code that mentions all the different layers. In the case of SqueezeNet, the layers are all defined in self.features(). Also, the number of output channels in the model is 512 as mentioned in the notebook

![](https://github.com/The-DL-Nerds/Global_Weed_Detection/blob/master/README_Img/3.PNG)

- Add the following lines of code to the template mentioned above:

```shell
class SqueezeFeatures(nn.Module):
    def __init__(self):
        super(SqueezeFeatures, self).__init__()
        base_model =  squeezenet1_0(pretrained=True)
        self.seq1 = nn.Sequential(base_model.features)
        self.out_channels = 512
    def forward(self, x):
        x = self.seq1(x)
        return x
backbone = SqueezeFeatures()
backbone.out_channels = 512
```

You now have added SqueezeNET as a backbone to the Faster RCNN model!

Here’s another example using GoogLeNet:

```shell
class GoogleFeatures(nn.Module):
    def __init__(self):
        super(GoogleFeatures, self).__init__()
        base_model =  googlenet(pretrained=True)
        self.seq1 = nn.Sequential(base_model.conv1,
                                  base_model.maxpool1,
                                  base_model.conv2,
                                  base_model.conv3,
                                  base_model.maxpool2
                                  )
        self.seq2 = nn.Sequential(base_model.inception3a,
                                  base_model.inception3b,
                                  base_model.maxpool3,
                                  base_model.inception4a,
                                  base_model.aux1
                                  )
        self.seq3 = nn.Sequential(base_model.inception4b,
                                  base_model.inception4c,
                                  base_model.inception4d,
                                  base_model.aux2
                                  )
        self.seq4 = nn.Sequential(base_model.inception4e,
                                  base_model.maxpool4,
                                  base_model.inception5a,
                                  base_model.inception5b
                                  )
        self.out_channels = 192
    def forward(self, x):
        x = self.seq1(x)
        return x
backbone = GoogleFeatures()
backbone.out_channels = 192
```

Besides SqueezeNet and GoogLeNet, we also experimented with DenseNet, ResNext and ran pre-defined backbones VGG-16 and ResNet 152 from the PyTorch library. Overall, implementing ResNext gave us the best accuracy of 66.24%

## Inference With Pseudo Labeling

### Inference

The competition guidelines required for us to run our inference notebooks offline. As a result, we could not run FRCNN with different backbones without downloading the files off the internet during runtime. To solve this problem, here is what we did:

- Copy the structure of the backbone you want to use from the PyTorch GitHub onto a new cell in Kaggle. This can be accessed in the model.py file found in vision/torchvision/models/. Here the model.py file could be any of the models in the models folder directory—inception.py, squeezenet.py, etc.

![](https://github.com/The-DL-Nerds/Global_Weed_Detection/blob/master/README_Img/4.PNG)

- Download the required .pth file from the models_urls dictionary by pasting the file link on to the web browser. Add this file to your personal Kaggle dataset.

![](https://github.com/The-DL-Nerds/Global_Weed_Detection/blob/master/README_Img/5.PNG)

- Next, change the following lines of code in the cell that you pasted the CNN structure:

  - Paste the file path in place of the url that you downloaded

![](https://github.com/The-DL-Nerds/Global_Weed_Detection/blob/master/README_Img/6.PNG)

  - Change: 

```shell
if pretrained:
    _load_state_dict(model, model_urls[arch], progress)
```
   To: 

```shell
if pretrained:
    model_urls[arch]
```
 - Remove “progress” from all the functions and return statements. Example:

From:

```shell
def densenet201(pretrained=False, progress=True, **kwargs):
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)
```

To:

```shell
def densenet201(pretrained=False, **kwargs):
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, **kwargs)
```

Make sure to do this for every model variation defined in the cell.

 - Turn “Internet” off and run the cell. The cell should run error free!

Note: Make sure to download the right .pth files and add the correct file path to the modified code. Forgetting to remove the “progress” from every single function can spring up errors in the code!

### Psuedo-Labeling

Psuedo labelling is a form of semi-supervised learning. We first train a model, as usual, on labelled images. This trained model is then used to predict and create labels for unlabelled images. They are also known as ‘pseudo labelled’ images. These pseudo labelled images are then mixed with already labelled images, and the model is retrained. 

![](https://github.com/The-DL-Nerds/Global_Weed_Detection/blob/master/README_Img/1.PNG)

This article from [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/) does a good job at explaining it. 

The inference code was largely derived from this notebook [FasterRCNN Pseudo Labeling](https://www.kaggle.com/nvnnghia/fasterrcnn-pseudo-labeling)
