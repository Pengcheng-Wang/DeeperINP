# BayesianDeeperINP
In this repository, we implement Bayesian NN following Gal's method, also incorporate implementations of deeper NNs (Hightway NN, RHN, ResNet, GridLSTM) in player simulation modeling and DRL in interactive narrative personlization.

This repository is built on the prior Ijcai DRL-INP repository, which built on Torch and related libs from Jan 2017.

When this repository starts, newer Torch, RNN and other corresponding libs have been applied. (Oct 2017)

Dec 9, I am trying to modify player outcome prediction as a regression problem. I think this is a good way to implement soft labeling.

Dec 20, 2017. I've tried to catch the performance of score prediction classification and regression modules. It seems like in most of the cases, these two modules make pretty similar conclusions. And incorrect predictions in validation set are very probable the case that when the prediction model are confident the data point should belong to one category, while the true lable is that it belongs to the other (with very high/low nlg). So, I suppose it will not be very helpful to utilize regression output to assist the prediction/classification decision.

Start to enhance the work from Jul 30, 2018. For dissertation purpose.
