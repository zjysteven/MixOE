# General structure
The assumed structure of this folder is as follows.
```
data
├── aircraft
│   ├── images
│   ├── ...
├── bird
│   ├── images
│   ├── ...
├── butterfly
│   ├── images_small
│   ├── ...
├── car
│   ├── cars_test
│   ├── cars_train
│   ├── ...
├── WebVision
│   ├── flickr
│   ├── google
│   ├── info
```


# FGVC datasets preparation
After you download and unzip each dataset, either rename them or make soft links such that everything follows the above structure.

Aircraft: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

Bird: https://dl.allaboutbirds.org/nabirds

Car: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

Butterfly: https://www.dropbox.com/sh/3p4x1oc5efknd69/AABwnyoH2EKi6H9Emcyd0pXCa?dl=0


# OE datasets preparation
WebVision: https://data.vision.ee.ethz.ch/cvl/webvision/download.html
- Download the resized small version of Flicker and Google images in WebVision 1.0
- Also download the "Training & Validation Labels", which corresponds to the "info" folder under "WebVision" in the structure shown above