# Project: Can you recognize the emotion from an image of a face? 
<img src="figs/CE.jpg" alt="Compound Emotions" width="500"/>
(Image source: https://www.pnas.org/content/111/15/E1454)

### [Full Project Description](doc/project3_desc.md)

Term: Spring 2020

+ Team 04
+ Team members:
	
1. Chen, Shuyi 
2. Flack, Jasmin 
3. Hong, Zidi 
4. Zhang, Ruo zhou 
5. Zhang, Xinlin 

+ Project summary: In this project, we created a classification engine for facial emotion recognition. The baseline model uses gradient boosting machine (gbm) with the accuracy of 42.4%. We have also evaluated other machine learning models(XGboost, Bagging-log,bagging-SVM) and chose the best one (bagging-SVM) based on predictive performance and computation efficiency. Our final advanced model uses bagging-SVM with PCA which enhances the accuracy rate by 52.56%.

+ Model performance:            
                      
GBM: The baseline model accuracy: 42.4% and time used on training the model is 805.146s.  
XGBOOST : The improved model Accuracy: 50.2% and time used on training the model is 483.013s.  
Bagging-SVM : The final advanced model with Accuracy : 52.25% and time used on training is about 110s.  
Bagging-Log : The improved model with Accuracy : 53.67%(Not stable) and time used on training(without five session) is 140s.



Note:
Our baseline model is more than 100M, so we presented in google drive and our advanced models are embedded in a ipynb.file(https://github.com/TZstatsADS/Spring2020-Project3-group4/blob/master/doc/Advanced%20Model%20and%20Other%20Models_colab.ipynb) which could use google colab to open. 

Steps:
1. Baseline model: https://drive.google.com/file/d/16ZQ-hkR1sJURZNX_NIpXcOsRSNsXgwyC/view?usp=sharing  
2. Go to this link: https://colab.research.google.com/notebooks/intro.ipynb
3. From the tabs at the top of the file explorer, select a source and navigate to the . ipynb file you wish to open. 


	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) 

All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement. 

+ Shuyi Chen built and trained the Xgboost model collaborated with Xinlin Zhang. The entire process is developed in Python including data prepossessing, train/test split, feature extraction(calculation of pairwise euclidean distances), hyperparameter tuning using gridsearch and model assessment. 
+ Zidi Hong was in charge of GBM model presented in a R Markdown file which contains data processing, train/test split, feature extraction, hyperparameter tunning and model assessment. 
+ Ruo zhou Zhang built and trained the Bagging-SVM model(Our advanced Model) as well as the Bagging-Logistic Model, including data prepossesing/feature extraction, hyperparameter tunning, train/test split and test. He also  wrap up the models in a certain function which can be simply called by the user. 
+ Xinlin Zhang built and trained the Xgboost model collaborated with Shuyi Chen. The entire process is developed in Python including data prepossessing, train/test split, feature extraction(calculation of pairwise euclidean distances), hyperparameter tuning using gridsearch and model assessment. 
+ Jasmin Flack completed the slides and presentation of the project.



Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
