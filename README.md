# Customer Analytics Project

## 1. Project Overview

"Olguita" is a stored located in Ilo, Peru. The owner of the store has been collecting some data from their customers; however, he has never paid too much attention to it. Due to the COVID-19 pandemic he had to close his store. Therefore, he had to move things online. He has recently read online that he is able to send customize messages to his clients using the data he has collected before. This will result on an increase of sales. Also he has read he is able to predict whether new customers will buy from him again. 

## 2. Data

For this project, 3 csv files will be used `segmentation data.csv`, `purchase data.csv` and `Audiobooks_data.csv`. All files can be located in the `Data` folder.

### 2.1 `segmentation data.csv`

* **ID**: It shows a unique identificator of a customer.
* **Sex**: Biological sex (gender) of a customer. In this dataset there are only 2 different options. 0 for female and 1 for male.
* **Marital Status**: Marital status of a customer. 0 for single person and 1 for a non-single (divorced/separated/married/widowed).
* **Age**: The age of a customer in years, calculated as current year minus the year of birth of the customer at the time of creation of the dataset. Min value is 18 and the max value is 76.
* **Education**: Level of education of the customer. 0 for "other/unknown", 1 for "high school", 2 for "university", 3 for "graduate school".
* **Income**: Self-reported annual income in US dollars of the customer. The min value is 35832 and the max value is 309364.
* **Ocupation**: Category of occupation of the customer. 0 for "unemployed/unskilled", 1 for "skilled employee/official" and 2 for "management/self-employed/highly qualified employee/officer".
* **Settlement size**: The size of the city that the customer lives in. 0 for small city, 1 for mid-sized city and 2 for big city. 

All values are integers. However not all values should be used as integers (e.g. **ID** should be used as a string). This dataset possess 2000 entries and 7 columns.

### 2.2 `purchase data.csv`

* **ID**: Shows unique identificator of a customer.
* **Day**: Day when the customer has visited the store.
* **Incidence**: Purchas incidence. 0 means the customer has not purchased an item from the category of interest. 1 means the customer has purchased an item from the category of interest. 
* **Brand**: Shows which brand the customer has purchased. 0 means no brand was purchased. 1,2,3,4,5 are the IDs of the other brands.
* **Quantity**: Number of items bought by the customer from the product category of interest. 
* **Last_Inc_Brand**: Shows which brand the customer has purchased on ther previous store visit 
* **Last_Inc_Quantity**: Number of items bought by the customer from the product category of interest during their previous store visit. 
* **Price_1**: Price of an item from Brand 1 on a particular day.
* **Price_2**: Price of an item from Brand 2 on a particular day.
* **Price_3**: Price of an item from Brand 3 on a particular day.
* **Price_4**: Price of an item from Brand 4 on a particular day.
* **Price_5**: Price of an item from Brand 5 on a particular day.
* **Promotion_1**: Indicator whether Brand 1 was on promotion or not on a particular day.
* **Promotion_2**: Indicator whether Brand 2 was on promotion or not on a particular day.
* **Promotion_3**: Indicator whether Brand 3 was on promotion or not on a particular day.
* **Promotion_4**: Indicator whether Brand 4 was on promotion or not on a particular day.
* **Promotion_5**: Indicator whether Brand 5 was on promotion or not on a particular day.
* * **Sex**: Biological sex (gender) of a customer. In this dataset there are only 2 different options. 0 for female and 1 for male.
* **Marital Status**: Marital status of a customer. 0 for single person and 1 for a non-single (divorced/separated/married/widowed).
* **Age**: The age of a customer in years, calculated as current year minus the year of birth of the customer at the time of creation of the dataset. Min value is 18 and the max value is 76.
* **Education**: Level of education of the customer. 0 for "other/unknown", 1 for "high school", 2 for "university", 3 for "graduate school".
* **Income**: Self-reported annual income in US dollars of the customer. The min value is 35832 and the max value is 309364.
* **Ocupation**: Category of occupation of the customer. 0 for "unemployed/unskilled", 1 for "skilled employee/official" and 2 for "management/self-employed/highly qualified employee/officer".
* **Settlement size**: The size of the city that the customer lives in. 0 for small city, 1 for mid-sized city and 2 for big city.

### 2.3 `Audiobooks_data.csv`

* **ID**: Shows unique identificator of a customer.
* **Book length(mins)_overall**: The overall book length is the sum of the lengths of purchases.
* **Book length (mins)_avg**: The average book length is the sum divided by the number of purchases.
* **Price_overall**: Overall price. 
* **Price_avg**: Average price.
* **Review**: It shows if the customer left a review (1 = yes).
* **Review 10/10**: It measures the review of a customer from 1 to 10.
* **Minutes listened**: Measure of engagement.
* **Completion**: It is the total minutes listened divided by the book lenght overall. 
* **Support Requests**: Total number of support requests (e.g. forgotten passwords to assistance).
* **Las visited minus Purchase date**: The bigger the better because it means a person is interacting with the platform.
* **Targets**: 1 if someone has bought again and 0 if not. 

## 3. Customer Segmentation

For this part of the project, the `1. Segmentation.ipynb` jupyter notebook will be used. 

The first step here is to drop any column which has a high correlation with another column. 

[correlations](Report/correlations.png)

The image above shows there is no high correlation between any columns. Therefore any column will be used.

### 3.1 Standardization
Now is time to apply `StandardScaler` because variables that are measured at different scales do contribute equally to the model fitting & model learned function and might end up creating bias. Other scaler methods could be used such as Normalization or Min-Max-Scaler.
`StandardScaler` uses the following formula: 
$z = \frac{x - mean}{std}$

### 3.2 PCA
PCA stands for Principal Component Analysis and it is the process of computing the principal components and using them to perform a change of basis on the data, sometimes using only the first few principal components and ignoring the rest. The key is to find now how many components will be needed. Therefore, a plot of cumulative variance of the components vs number of components will be displayed. The variance here means the amount of information it holds. The number of components which cumulative sum reaches 80% will be used.  

[explained_variance](Report/explained_variance_components.png)

The graph above shows that 3 components will be used. Now it is time to find the relation of the components with the other features. 

[compoent 1](Report/component_1.png)

Component 1 has a positive correlation with income, occupation, age and settlement size. This component relates to the career of a person.

[compoent 2](Report/component_2.png)

Component 2 has a positive correlation with education, marital status and sex. This component relates to the individuals education and lifestyle. 

[compoent 3](Report/component_3.png)

Component 3 has a positive correlation with age and negative correlation marital status and occupation. This component relates to experience of a person.

### 3.3 K-Means

K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data. The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. For this project, we need to find the most optimal value for K. To achieve this, we will run a loop and try different K values, in order to assess the optimal K value, inertia will be extracted. Inertia can be recognized as a measure of how internally coherent clusters are. The model will be fitted with the standardize data applying PCA. A plot using the number of clusters in the x-axis and inertia in the y-axis will be displayed. 

[kmeans](Report/inertia_numberclusters.png)

In the graph above, the elbow method was applied. In cluster analysis, the elbow method is a heuristic used in determining the number of clusters in a data set. The method consists of plotting the explained variance as a function of the number of clusters, and picking the elbow of the curve as the number of clusters to use. In this case K=4 is the best way to go. 

After applying kmeans to the data, the following summary is obtained:

[kmeans summary](Report/clusters_summary.png)

The table above shows that "cluster 0" has 265 observations a represents 13.25% of our observations. "Cluster 0" is both high in component 1 and 2 which means high in education, lifestyle and career. This cluster will be called "well-off". "Cluster 1" shows that it has 460 observations which represent 23% of the data. "Cluster 1" is both low in component 1 and component 2 which means low in education, lifestyle and career, "cluster 1" will be called "fewer-opportunities". "Cluster 2" has 692 observations which represents 34.6% of the data. "Cluster 2" is low in component 1 (career) and high in component 2 (education and life-style). "Cluster 2" will be called "standard". "Cluster 3" has 583 observations which represents 29.15% of the data. "Cluster 3"is high in component 1 (career) and low in component 2 (education and life-style). This cluster will be called "career-focused".

The following graph will show how well k-means perform:

[3D kmeans](Report/customer_segmentation.png)

The models used to standardize, apply PCA and to clusters will be saved in the `Segmentation Models` folder with the following names:
1. `scaler.pickle` 
2. `pca.pickle`
3. `kmeans_pca.pickle`

## 4. Descriptive Analytics

The three models `scaler.pickle`, `pca.pickle` and `kmeans_pca.pickle` are used in the dataset `purchase data.csv` to predict to which cluster belongs each observation. `purchase data.csv` contains data for chocoloate different prices and brands, whether a customer has buy it or not, the quantity of purchase, whether the customer has access to a type of promotion or not, and many other things.  

The following pie chart shows the segment percentages.

[segment percentages](Report/ds_segment.png)

The chart depicts that 37.8% of observations belong to the "fewer-opportunities" segment, the 19.6% of observations belong to the "well-off" opportunities, the 20.6% of observations belong to the "standard" segment and that 22% of observations belong to the "caree-focused" segment. 

Then some descriptive statistics is needed to apply for each segment. The average number of visits for each segment is shown below:

[visits segment](Report/ds_visits_segment.png)

The bar graph shown above shows that the average number of visits for the "well-off" segment is 117.30. The average number of visits for the "fewer-opportunities" segment is 113.73. The average number of visits for the "standard" segment is 117.70. The average number of visits for the "career-focused" segment is 123.45.

The average number of purchases for each segment is shown below:

[purchases segment](Report/ds_purchases_segment.png)

The bar graph shown above shows that the average number of purchases for the "well-off" segment is 34.60. The average number of purchases for the "fewer-opportunities" segment is 22.76. The average number of purchases for the "standard" segment is 24.90. The average number of purchases for the "career-focused" segment is 39.83.

The average purchases per visit for each segment is shown below:

[purchase visit segment](Report/ds_purchase_visit_segment.png)

The bar graph shown above shows that the average purchases per visit for the "well-off" segment is 0.28. The average purchases per visit for the "fewer-opportunities" segment is 0.20. The average purchases per visit for the "standard" segment is 0.21. The average purchases per visit for the "career-focused" segment is 0.28.

Brand choice for each segment is also important to know. The graph below shows the findings:

[brand choice](Report/ds_brand_choice.png)

It can be seen that the "well-off" segment has a preference for brand 4, the "fewer-opportunities" segment has a preference for brand 2, the "standard" segment don't really have a preference for a given chocolate, maybe for brand 2, and finally the "career-focused" segment has a preference for brand 5.

Finally it is desirable to show who much revenue each brand brings, the graph below shows the findings. 

[revenue brands](Report/ds_revenue_segment.png)


## 5. Deep Learning for Convertion Prediction

The goal of this part of the project is to use deep learning to predict whether a customer will purchase again or not.

### 5.1 Data Preparation

The data 11847 observations for target 0 and 2237 observations for target 1. The data needs to be balance, therefore, first we need to shuffle the 11847 observations for target 0 and then take a sample of 2237 observations, in that way we have for both targets the same number of observations. 

The second step in preparing the data is that we need to standardize the data so that our algorithm is unbiased towards different features. We will use `StandardScaler` from `sklearn` for this purpose. 

The the data will be converted in tensors and DataLoaders so that we can use batches to run our algorithm. 

### 5.2 Model

Our model architecture consist of 5 layers. Our input layer will have 10 units, are first hidden layer will have 50 units, our second hidden layer will have 100 units, our third hidden layer will have 50 units and our output layer will have 2 units. We will use elu as our activation function for all layers and we will use dropout with a probability of 20% in order to avoid overfitting. Our criterion will be `CrossEntropyLoss` however, for future work, experimenting with `BCEWithLogitsLoss` will be a good idea since this is a binary problem. The `Adam` optimizer was used.  

### 5.3 Training

50 epochs were used and an valid accuracy of 82% was obtained. 

## 6. References

* https://towardsdatascience.com/how-and-why-to-standardize-your-data-996926c2c832#:~:text=StandardScaler%20removes%20the%20mean%20and,standard%20deviation%20of%20each%20feature.
* https://www.codecogs.com/latex/eqneditor.php
* https://stackoverflow.com/questions/11256433/how-to-show-math-equations-in-general-githubs-markdownnot-githubs-blog
* https://en.wikipedia.org/wiki/Principal_component_analysis
* https://www.kaggle.com/shrutimechlearn/step-by-step-kmeans-explained-in-detail
* https://en.wikipedia.org/wiki/Elbow_method_(clustering)#:~:text=In%20cluster%20analysis%2C%20the%20elbow,number%20of%20clusters%20to%20use.
* https://scikit-learn.org/stable/modules/clustering.html
* https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
* https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html
* https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets

**Note** Special thanks to 365DataScience for the data and the lessons given.