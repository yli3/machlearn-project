# Qualitative Human Activity Recognition via Supervised Learning

## Introduction

As part of the "quantified self" movement, the gathering of personal activity data is becoming more and more ubiquitous. An emerging frontier is the challenge of not merely quantifying, but qualifying "how well" an activity is performed. *Velloso et. al. (2013)* provides a raw dataset of activity sensor readings from a worn cellphone for users performing biceps curls in one of five different ways. We use supervised learning to tackle the classification problem: given a set of indicators, can we build a model that predicts the correct "class" of bicep curl performed?

## Approach
There are a broad range of methods available to us that fall under the broad field of supervised machine learning. We say it is *supervised* because we make use of a provided set of answers -- that is, the knowledge of a certain set of indicator data empirically known to result from a certain categorical outcome. We call it *learning* because we use this provided data to inform and create a model that is able to predict outcomes for general, future indicator data.

All models are wrong (George Box, 1976), but some may be useful. To quantify usefulness, we attempt to estimate **out of sample** error, or the error rate of the prediction model on new data. This quantity can only be estimated, as not all new data can be known.

To estimate this error, we split the provided data into separate **train** and **test** sets; the former to be used exclusively for building the model, and the latter exclusively to estimate model quality.

## Background

## Data exploration

## Decision Trees


## Random Forest
We select the Random Forest method as it is a combination approach that is well suited to categorization problems with numerous indicator variables.
