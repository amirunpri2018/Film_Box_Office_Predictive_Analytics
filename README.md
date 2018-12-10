# Film Box Office Prediction Analytics

<center>Columbia University IEOR4523  2018 Fall  Data Analytics Course Project</center>



## Overview

### Introduction

As movies have become important part of our daily lives, people go to see movies for various reasons. Some go for their favorite actors, some go because of other people’s good reviews, and some just go for fun. As a group, we think that there is a diversity of potential factors that can significantly contribute to a hit movie, so we  this research, utilizing data analytical methods including sentimental analysis and machine learning prediction.

In this Github directory, *Report* file is our final presentation PPT at Columbia course, *Scraping* contains py file for accessing both IMDb and Rotten Tomatoes reviews, *SentimentAnalysis* refers to the film review sentimental analysis and *Visualization* has the data analysis component with various explorative data visualization. 

For detailed information, please find the Summary pdf file, which has a top-bottom introduction to the whole project; *DataDescription.docx* could give you a general review of the total data. For short version, excerpts of major results are highlighted below. Also note we preserve the data file and do not upload them online.  

Last but not the least, this **team project** is contributed to the **great joint collaboration** of our [fantastic team](# Cast) with Qinya, Huizi, Xiaoxuan, Xiaoyu and Xiaohui.   



### Data Source

Web Scraping Source: Rotten Tomatoes & IMDb

Period: 2014 – 1st half of 2018

Total Amount of Movies: 2864

Fields: Genre, Release Date, Studio, Budget, Open Weekend Revenue, Gross Revenue, Actor, Director, Writer, Meta-Score, Vote, Reviews from Critics, Reviews from Audience

### Data Analysis

#### Feature Description

After gathering and filtering the data, we analyzed data to find some interesting patterns and provide useful insights. 

* Genre

* Open Week Revenue & Gross Revenue

* Actor, Director & Studio

* Release Date

* Reviews from critics and audience

####  Sentimental Analysis

We focus on analyzing the general sentiment towards films with text review, specifically along two dimension over Rotten Tomatoes’ critic review and IMDb’s audience review. 

We implemented a two-layer Bidirectional LSTM and the performance results is shown below:

| **Metrics** / **Dataset** | **Rotten Tomato Critics** | **IMDb Users** |
| ------------------------- | ------------------------- | -------------- |
| **Precision**             | 67%                       | 57%            |
| **Recall**                | 64%                       | 46%            |
| **F-Score**               | 65%                       | 48%            |
| **Accuracy**              | 86%                       | 88%            |

### Conclusion

#### a. Quantitative

We test and evaluate four models on the test set, XGBoost is the best in prediction with F1 Score of 76%, the result is illustrated below:

| **Metrics/Models** | **Naïve   Bayesian** | **Neural   Network** | **SVM** | **XGBoost** |
| ------------------ | -------------------- | -------------------- | ------- | ----------- |
| **Precision**      | 67%                  | 57%                  | 84%     | 77%         |
| **Recall**         | 64%                  | 46%                  | 43%     | 76%         |
| **F1-Score**       | 65%                  | 48%                  | 51%     | 76%         |
| **Accuracy**       | 86%                  | 88%                  | 82%     | 75%         |

#### b. Qualitative

Big-deal sci-fiction movies flooding in during the summer vacation bring a feast for eyes to the audience, though not always acquiring corresponding praises. Instead, during October and November, the grand exhibition of Oscar-level hit movies are likely to make you a surprise. 

What matters for a movie investor is the combo of famous director with all-star actors which contributes to a guarantee of considerable box office revenue. Nevertheless, open-week marketing is never less important as for promoting the earnings. At length, what equips a movie with the sustainability to survive the time is of course, its quality.



# Cast

- [Qinya Li](https://github.com/qinya1)

- [Xiaohui Li](https://xiaohui-victor-li.github.io/)

- [Xiaoxuan Xia](https://github.com/XiaoxuanXia)

- [Xiaoyu Xu](https://github.com/xx2302)

- [Yanghuizi Wang](https://github.com/YanghuiziWang)

