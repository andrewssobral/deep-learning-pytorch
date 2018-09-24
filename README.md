## PyTorch Deep Learning Models

Last page update: **24/09/2018**

This repository aims to summarize and standardize some deep neural networks implemented in PyTorch for image classification, detection and segmentation. Other applications like background subtraction, object tracking are also welcome.

Note: *This repository is a work in progress, and significant changes are expected.*

**List of available models:**

* Image Classification

* * *...in progress*

* Image Object Detection

* * *...in progress*

* Image Segmentation (multiclass and binary segmentation)

* * FCN8, FCN16, FCN32
* * SegNet
* * PSPNet
* * UNet

Note:
* Some models differs from the original implementation. 
* SegNet does not match original paper performance.
* PSPNet misses "atrous convolution".

*Other kinds of models are welcome!*

**Requirements**
* pytorch >=0.4.0 (some parts of the code work on 0.3.x)
* torchvision >=0.2.0
* visdom >= 0.1.8.5 (optional, for loss and results visualization)

**Motivation**

After the first release of PyTorch, many people, such as scientific researchers, students, and professionals, worked hard to implement several state-of-the-art deep learning models for several AI applications, such as computer vision, natural language processing, [among others](https://machinelearningmastery.com/inspirational-applications-deep-learning/). They also helped the deep learning community by open sourcing their algorithms implementation, enabling an exponential growth in the number of source codes available on the Internet, as you can see in the following links:

* https://github.com/ChristosChristofidis/awesome-deep-learning
* https://github.com/kjw0612/awesome-deep-vision
* https://github.com/amusi/awesome-object-detection
* https://github.com/mrgloom/awesome-semantic-segmentation
* and more...

**The problem** Here's our motivation! :)
Even with this massive list of algorithms and their respective source code, it's difficult to find something clean, simple, easy-to-use, and, `the most important`, that's run on the first try, even on CPU or GPU, Windows, Mac or Linux, etc. If you tried to run several deep learning models from different authors, you understand me :)
So, the main idea behind this project is to ensure the **simplicity**, **interoperability** and **standardization** of the current available deep learning models, by doing code refactoring and asserting a clean, simple and easy-to-use code format. 

Everybody is invited to join this initiative and colaborate with us by sending pull requests. We highly appreciate simple, clear, and easy-to-understand source codes. If you find something wrong in this repository, or something that could be improved, let's do it and send us your pull requests ;)

This project was ispired by the following GitHub projects:

* https://github.com/bodokaiser/piwise
* https://github.com/meetshah1995/pytorch-semseg
* https://github.com/zijundeng/pytorch-semantic-segmentation

Many thanks for the authors for this great initiative!

**License** 

Source codes, algorithms implementation, and everything provided in this repository is considered **free** for personal, academic, and commercial purposes.

**About the author**

I'm a Senior Software Engineer (15+ years) with a Ph.D. in Computer Vision and Machine Learning. 

* Personal home page:
http://andrewssobral.wixsite.com/home

* Twitter: http://twitter.com/andrewssobral
