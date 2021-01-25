# LettuceWrap
## LCA Wrapper for the Tensorflow 2.4 Keras Model Class

The goal of this project is to make it simple and seamless to integrate the loss change allocation metric as created by Uber AI (https://eng.uber.com/loss-change-allocation/).

The class takes an additional keyword argument, layer_names. This object should be a list of strings coinciding with the layers of your model which you want to calculate the lca for.

A second frontend modification is the LCAWrap.Fit() method in contrast to Model.fit().
The Fit() method will calculate and store(or stream) the LCA at the end of each epoch, and makes calls to the the Model.fit() method for the actual training. As such the fit() method still exists within LCAWrap with no modifications to the functionality.




