#incubator 
###### upstream: [[Deep Learning]]

### Origin of Thought:
- let's think of a non-mathematical way of thinking about convolution

### Solution/Reasoning: 

**Binoculars Analogy**

Let's think of an image as a big, bustling city viewed from the sky, and the features in the image (like edges, shapes, colors, etc.) as various important places in the city - parks, buildings, landmarks and so on.

Now, imagine you're in a hot air balloon with a pair of binoculars, but you can only look straight down. You want to make sense of the city, but you can only see a small area through your binoculars at a time.

In this scenario, the act of convolution is like using your binoculars to scan across the city, one patch of land at a time, to map out important features. When you're looking through the binoculars, you are focusing on a small area - this is similar to the "filter" or "[[Kernel]]" in convolution. This filter is responsible for finding certain features, like recognizing water bodies or green parks.

As you move your binoculars across the city (or the filter moves across the image), you're creating a new map (or feature map) that emphasizes the things you're interested in. Each spot on your new map has a value that says how much the area you just looked at resembles the feature you're interested in. So, if your binoculars (filter) are good at spotting water bodies, the new map you create will highlight all the bodies of water in the city very well.

That's the core idea of convolution in the context of **Convolutional Neural Networks**. A filter moving over an image, focusing on one small area at a time, and creating a new matrix that emphasizes certain features from the original image.

*but how do we set the kernels to begin with?*

Let's answer this by talking about setting the parameters in **tensorflow**

The `kernel_size` parameter defines the height and width of the 2D convolution window, which is equivalent to the area your binoculars can see at a time in your analogy. This is usually set to (3, 3) or (5, 5), which means the filters will consider 3x3 or 5x5 pixel areas at a time.

`filters` specifies the number of different kernels to be applied. This can be interpreted as the number of different "features" you want your model to be capable of recognizing. For example, if you're analyzing colored images and you set filters to 64, the model will learn 64 different ways of transforming the input data, like detecting various types of edges, curves, colors, etc.

`strides` specify the steps by which the window (or your binoculars in your analogy) moves across the image. A stride of (1, 1) moves the window one pixel at a time. Increasing the stride would mean the window skips some pixels which would result in a smaller output size.

`padding` deals with the border of your image. If set to 'valid', it means no padding, i.e., the convolution is only computed where the input and the filter fully overlap, and thus the output size might be smaller. If set to 'same', it means the output has the same width and height as the original input, achieved by padding the input in such a way that the filter "fits" the input data nicely.

`activation` is the activation function to use after the convolution. The activation function introduces non-linearity into the model. Common choices are 'relu' (Rectified Linear Unit), 'tanh' (Hyperbolic Tangent), or 'sigmoid'.

`input_shape` is the shape of the input data you're feeding into the model. You usually only need to specify this for the first layer of the model. If your images are 64x64 RGB images, this would be (64, 64, 3), because the images are 64 pixels wide, 64 pixels high, and have 3 color channels.

In TensorFlow, you might set up a convolutional layer like this:

```python
layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(64, 64, 3))
```

This creates a convolutional layer with 32 filters of size 3x3, a stride of 1 (in both dimensions), padding that preserves input dimensions, and the ReLU activation function. This layer expects input tensors to be 64x64 with 3 channels.

In terms of how you specifically choose these parameters, it largely comes down to experience and testing. Different tasks will require different architectures and parameter choices, and part of building good models is experimenting with these parameters to see what works best for your specific task. There are some general rules of thumb (like using small kernels of 3x3 or 5x5, and using 'relu' as your activation function), but it will ultimately depend on the specifics of your task.