# Kaggle Diabetic Retinopathy Solution.
Fifth place solution for the [Kaggle Diabetic Retinopathy competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/) including some trained network. For more information see my [blog post](http://jeffreydf.github.io/diabetic-retinopathy-detection/).

The code is quite badly written (lots of ideas, not so much time) but I figured it was more important to release it quickly than to rewrite it all and only publish it weeks later. The most interesting file will probably be the notebook at 
[notebooks/Sample_prediction.ipynb](https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/master/notebooks/Sample_prediction.ipynb) which shows how to load the model dump, do some predictions with it and see the activations for the different layers.

Important to note: the code is built on top of Lasagne at commit cf1a23c21666fc0225a05d284134b255e3613335. My [own fork](https://github.com/JeffreyDF/Lasagne) currently uses that version so for a little while that will still work. For Theano I have been using the latest master and have no problems.
