Tensorflow implement & optimize for Custom Logo Detection base on Faster-RCNN VGG16 Architecture
[Paper](https://arxiv.org/abs/1506.01497)
* Run demo
  - Download pretrained model <VietinBank, VietcomBank, BIDV> : [GDrive](https://drive.google.com/open?id=1xQDOM7qXln-sp4-rEMReWkFJscHfmKi2)
  - Download some test images [Options]: [GDrive](https://drive.google.com/open?id=1CIExYmXyfvnq2VRkjrqS7hg13F0-QptO) 
  - Install system requirements: 
    ```
    pip3 install -l requirements.txt
    ```
  - Run UI: 
    ```
    pip3 LogoDetectionUI.py
    ```
* Have done:
  - Reduce about 40% parameters 
  - Increase precision & recall
  - Increase 2x performance
