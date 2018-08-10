# Semantic Segmentation
    
[//]: # (Image References)
[image1]: ./images/fcn_vgg.png
[image2]: ./images/semantic_segmentation.gif
[image3]: ./images/1.png
[image4]: ./images/2.png
[image5]: ./images/3.png
[image6]: ./images/4.png    

### Model Architecture
In this project, I labeled the pixels of a road in images using a Fully Convolutional Network (FCN). I use transfer learning from a pretrained VGG16 model. Kitti road dataset is used for training the model.

![][image1]

FCN model is based on a pretrained VGG-16 model include input layer, keep probability layer, layer 3, layer 4 and layer 7. Convolutional 1x1 of vgg layer 7 to mantain space information. layer 7 is connected to convolutional layer with kernel size = 1 and then Upsample deconvolution x 2 with kernel = 4, stride = 2. 1x1 convolution of vgg layer 4 is also connected to convolutional layer with kernel = 1. The above two output layers are summed up to form the first skip layer. The first skip layer is Upsample and connected to convolutional layer with kernel = 4 and stride = 2. The output of 1x1 convolution of vgg layer 3 connecting to convolutional layer with kernel = 1 summed up the above output layer form the second skip layer. The second skip layer is connected toUpsample deconvolution x 8 with kernel = 16, stride = 8 to form the final output layer. All convolution and deconvolution layer using kernel initializer with standard deviation 0.01 and L2 regularizer 0.001.    

The inference results demonstrate the green regions as the road as below: The model architecture is as follow:
    
![][image2] 

##### Optimizer
  - Adam optimizer 
  - Cross-entropy loss function.    
  
#### Hyperparameter   
Learning rate is fixed at 0.0001 and keep probability is set to 50%.
    
#### Inference Result
    
![][image3]
![][image4]
![][image5]
![][image6]

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
