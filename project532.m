addpath /Users/kaeda/Documents/uw/2015Fall/CS532/project/SMNI_CMI_TRAIN_features
load eeg_labels4.mat
y=aaa;
load entropy.mat
load features4_noentropy.mat
%%
 y(11:20)=zeros(10,1)';
[Beta lassoinfo]=lassoglm(features,y,'binomial','CV',10);