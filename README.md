# Titanic - Machine Learning from Disaster

## Introduction

This project is the 'Hello World' of ML and one of the most famous Kaggle competitions.

[Link to Kaggle website where there is an overview of the project and of the data](https://www.kaggle.com/competitions/titanic/overview)

The goals of this project is to use machine learning to create a model that predicts which passengers survived the Titanic shipwreck. Normally, we would
have two datasets with Titanic passengers, the 'train.csv' and the 'test.csv', the former explore is used to explore the data, search for motifs  and train
our model, while the latter is used to evaluate our model.

In this project we do not take part to the competition, thus we use only the 'train.csv' data (where it is known if a passenger survived or not) to train 
and evaluate our model.

## Data Description

Survival: 0 = No, 1 = Yes

Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)

Sex: Sex of the passenger (female/male)

Age: Age in years (if the age is fractional, it is less than 1, if the age is estimated, it is in the form of xx.5)

SibSp: # of siblings (brother/sister/stepbrother/stepsister) or spouses (husband/wife) aboard

ParCh: # of parents (mother/father) or children (daughter/son/stepdaughter/stepson) aboard

Ticket: Ticket number

Fare: Passenger fare

Cabin: Cabin number

Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Code Overview

1. We read the data and perform extented exploratory data analysis

2. We perform feature engineering in order the replace missing data and create new useful variables 

3. We create different models in order to predict if a passenger survived the shipwreck or not

4. We compare the different models and comment on their performance
