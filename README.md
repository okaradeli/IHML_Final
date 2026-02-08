IHML: Incremental Heuristic Meta-Learner classification Algorithm

Some tips to make this work: 
1- edit globals.py: Make sure the directories and data folders are correctly set. 
2- edit globals.py: Enable some BASE algorithms ( i.e. RF, XGB...) to be included in the meta-learner. 
3- run: incremental_metalearner , it will automatically start training/testing cycles

Note: that data folder contains partial samples, you need to get the whole dataset from source ( UCLA, etc)
 

The reference paper:
https://www.tandfonline.com/doi/full/10.1080/08839514.2024.2434309
