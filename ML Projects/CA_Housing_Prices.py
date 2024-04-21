#the following csv file was extracted from a tgz file


#now let us read the csv file into the program 
import pandas as pd



housing = pd.read_csv(r'C:\Users\thatm\Documents\PythonProjex\Python Projects AI ML\housing.csv')


#now lets go to the head 

housing.head()

#there are ten attributes in this file 

#they are: longitude, latitude, housing median age, total rooms, total bedroooms, population, households, median income, median house value, ocean proximity 

housing.info() #we use this info function to get a quick descriptioin of the data in particular the total number of rows, and each attribute's type and number of non null values 
    #ANALYSIS OF THE ATTRIBUTES 
    #there are 20,640 entries in this file. This means that this a fairly small dataset! 

    #while all the other attributes have 20,640 entries. the total bedrooms attribute has 20,433 entries. This means that 207 entries are missing this feature 

    #we see that ocean proximity variable is an object value, this means that it could be a text value 
    # we need to see what categories exist and how many districts belong to each category by using the value_counts() method: 


housing["ocean_proximity"].value_counts()

#let us also look at other fields, such as the describe method that shows a summary of the numerical attributes 

housing.describe()

#let us build histograms so that we can see what our dataset looks like 

#libraries that need to be imported for this 

import matplotlib.pyplot as plt 

housing.hist(bins=50, figsize=(20,15))
plt.show()


#things to note: 
    #median income not in USD.
    #after confirming this, the attribute has been capped at 15.0001 for higher median incomes and .4999 for lower median incomes 
    #many of these attributes have different scales (more on that later)
    #histograms are tail heavy 
        # when they have a heavier distribution on the right side of the histogram 
            #this is a problem in data because it can make it hard for machine learning algorithms to actually detect patterns. You can transform these attributes later on to have more bell shaped distributions 



#!!!!!!! BEFORE DOING ANYTHING ELSE IT IS NECESSARY TO CREATE A TEST SET AND PUT IT ASIDE !!!!!!!!!!!!!!!!!!!!!!!!!!!!


#creating a test set: just pick 20% of random instances. and then set them aside 


import numpy as np 
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")



import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] <256 *test_ratio

def split_train_test_by_id(data,test_ratio,id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

#the housing dataset does not have an identifier column: the simple solution is to use the row index as the ID

housing_with_id = housing.reset_index() #add an index column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


#note: you need to make sure that if you use row index as a unique identifier that the new data get appended to the end of the dataset and now row ever gets deleted

#if you cannot for some reason then you can try to use the most stable features to build a unique identifier. 
#for our example using the district's latitude and longitude are guaranteed to be stable 

#so you can also combine them into an ID 
    #for example:

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


#okay we will now use ScikitLearn 

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size= 0.2, random_state = 42)

#next up we will talk about sampling bias

#as we look into the median income we realize that the histogram flows around 2-5 (tens of thousands of dollars) but it could be true that some median incomes goes far beyond 6.

#its important to have enough instances in your DS for each stratum or your estimate of each stratums could be biased
    #this means you should not have too many strata and each stratum should be large enough. 

#the code below creates an income category attribute by dividing the median income by 1.5   
    #[this is to limit the number of income categories \]


housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5, 5.0, inplace = True)


#lets explain what was done above:
#we have created an income attribute so that the median income is divided by 1.5 
    #using ceil we have discrete categories 

#lastly we have merged every category greater than 5 into category 5 


#now we can do stratified sampling using scikit learn stratified shuffle split 

from sklearn.model_selection import StratifiedShuffleSplit


split = StratifiedShuffleSplit(n_splits= 1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]



#lets look at the income category proportions in the DS 

housing["income_cat"].value_counts()/len(housing)


#now we remove the income_cat attribute so that the data can go back to its original state 

for set in(strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis = 1, inplace = True)


#many of the ideas that we have spent time on are ideas that we will use for cross validation.


##########################################################################################

#  now its time to move on to the next stage: exploring the data


#first make sure you have put the test set aside and you are only exploring the training set.

#also if the training set is very large you may want to sample an exploration set. But this wont be necessary because our ds is very small. So we can go ahead and work with the full set. 

#lets create a copy so that it is possible to play with it without harming the training set:

housing = strat_train_set.copy()



#Visualizing Geographical Data:

#because there is geo info create scatterplot of all the districts and visualize the data 

housing.plot(kind = "scatter", x = "longitude", y = "latitude")
plt.show()

#while this looks like california but it is difficult to see any patterns. Setting the alpha to 0.1 makes it much easier to visualize high density data points 

housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha =0.1)
plt.show()

housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.4, s = housing["population"]/100, label = "population", c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True,)
plt.legend()
plt.show()


#from the image we have generated above we can see that it is useful to incorporate a clustering algorithm to detect main clusters and add new features that measure the proximity to the cluster centers



#looking for correlations: 

#Since the dataset is not to large, let us compute the standard correlation coefficient (Pearson's r), between every pair of attributes using the corr() method:

housing.info()

#let us drop the ocean proximity column because we are unable to create a corr_matrix or calculate correlation with it 
housing = housing.drop('ocean_proximity', axis =1 )
housing.info() #verify that ocean_proximity is dropped and it is 
corr_matrix  = housing.corr() #able to create matrix now 

corr_matrix["median_house_value"].sort_values(ascending = False) #this will calculate how each attributes correlates with the column mentioned in the line of code

#definition of the values: closer to 1 = positive correlation, -1 = negative correlation, 0 = no linear correlation

#WARNING: correlation coefficient only measures linear correlations. IT CAN COMPLETELY MISS OUT ON NONLINEAR RELATIONSHIPS 

#another way!!!!!!!!!!!!!!!
#use pandas' scatter_matrix function which plots every numerical attribute against every other numerical attribute


#let us do this now:

from pandas.plotting import scatter_matrix 
#this module is not found! find out how you can download it 
    #error fixed just had to take out the tools portion of it 

#while we could plot every attribute let us focus on the attributes that are correlated with mean housing value

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12,8))
plt.show()

#the most promising attribute looks lke median income so let's zoom in on their correlation scatterplot 

housing.plot(kind="scatter", x = "median_income", y="median_house_value", alpha =0.1)
plt.show()

#what do we see? 
#we see that the correlation is very strong 
#we can see an upward trend
#points are not too dispersed
#price cap is visible as a horizontal line at $500k
    #but also there are straight lines in the $450k and 350k lines

#one last thing you may want tod o before actually preparing the data for machine learning algorithms is to try out various attribute combinations.

#for example: the number of room in a district is not very useful if you dont know how many households there are. 
    #you want to know the number of rooms per household 

#similarly the total number of bedrooms you want the total number of rooms

#population per household also seems like an interesting attribute

#so let us do thattt

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

#now let us look at the correlation matrix again 
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)


#it is time to prepare the data for machine learning algorithms 

#we wont be doing this manually, we should learn how to write functions

#the reasons for this are as follows:

#this will allow you to reproduce these transformations easily on any dataset 
#you will gradually build a library of transformation functions that you can reuse in future projects

#you can use these functions in your live system to transform the new data before feeding it to your algorithm 

#this will make it possible for you to easily try various transformations and see which combination of transformations works best 


#let's revert to a clean training set by copying (strat_train_set once again)

#let us also separate the predictors and the labels since we don't necessarily want to apply the same trasnformationis to the predictors and the target values (note that drop()) creates a copy of the data and does not affect strat_train_set

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


#DATA CLEANING 



#note: there are some missing features for total_bedrooms let us fix this. 

#for this we have the following options: 
    # get rid of corresponding districts
    #get rid of the whole attribute 
    # set the values to some value (zero, the mean, the median, etc)


#hence we should use the dropna(), drop() and fillna()


housing.dropna(subset=["total_bedrooms"])  # get rid of corresponding districts
housing.drop("total_bedrooms", axis =1)#get rid of the whole attribute 
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median) # set the values to some value (zero, the mean, the median, etc)

#note if you are using option 3. The value needed to compute the median value should be the training set one. Additionally do not forget to store this value somewhere. This will late be needed to replace the missing values in the test set when you are evaluating your system. 



#Scikit-learn uses Imputer. 

#use: create an imputer instance. this will specify that you are trying to replace each attribute's missing values with the median of that attribute

from sklearn import impute

from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy="median")

#note: the median can only be computed on numerical attributes we need to create a copy of the data without the text attribute ocean_proximity: 

housing_num = housing.drop("ocean_proximity", axis = 1)

#lets use the imputer instance to the training data using fit()

imputer.fit(housing_num)

imputer.statistics_

housing_num.median().values

X = imputer.transform(housing_num)
X #this is an array lets put it back into a dataframe

housing_tr = pd.DataFrame(X, columns = housing_num.columns)

#remember we left out the ocean_proximity variable because it contained text attributes. We prefer to work with numbers - lets convert text to numbers 

    #the method that Scikit-Learn would use for this task LabelEncoder 

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded

print(encoder.classes_)

#Representation in ML Algorithms: the algorithm will sometimes assume that two nearby values are more similar than two distant values. So, this is an issue that we have to fix: 
    #create one binary attribute per category: 
    # one attribute equal to 1 when cat. is  1H Ocean (and 0 otherwise)
    # one attribute equal to 1 when cat is Inland (and 0 otherwise) 

# the method mentioned above is called one-hot encoding because only one attribute will be equal to 1 (hot) while the 0 (cold)
# 
#Scikit-Learn provides a OneHotEncoder to convert integer categorical values into one-hot encoders. 
# Note: fit_transform expects a 2d array but housing_cat_encoded is a 1D array so reshaping is needed 
# 
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1)) #allows one dimension too be -1 which means unspecified. The value would be inferred from the length of the array and the remaining dimensions 


housing_cat_1hot #output was a sparse matrix instead of a numpy matrix - this is useful since a lot of the contentsin the hot encoding would be full of zeros except for one row. This would use up a lot of memory. Sparse matrix only stores the location of the nonzero elements. This can mainly be used like a normal 2d array. But this could be converted into a dense NumPy array, just call the toarray() method: 

housing_cat_1hot.toarray()

#this transformation of text categories to integer categories an then from integer categories to one-hot vectors in one shot using the LabelBinarizer class

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)

housing_cat_1hot # this will return a dense numpy array by default. You can get a sparse matrix instead by sparse_output = true to the LabelBinarizer constructor

housing

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin): 
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self 
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/X[:,household_ix]
        population_per_household = X[:, population_ix]/X[:, household_ix]
        if self.add_bedrooms_per_room: 
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
            bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values) #there is one hyperparameter add_bedrooms_per_room set to true by default.
#this hyperparameter will allow you to easily find out whether adding this attribute helps the ML Algorithm.
# adding a hyperparameter to gate any data prep step that there is uncertainty with. 
# the more these data prep steps are automated the faster you will find a combination 



######################################################################################

#TRANSFORMATION PIPELINES

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])


housing_num_tr = num_pipeline.fit_transform(housing_num)


#pipeline constructor takes in list of name/estimator pairs defininf a sequence of steps. All but the last estimator must be transformers (they must have a fit_transform() method )


#StandardScaler applies all the transforms to the data in sequence ()

#is there a way to join pipeline and LabelBinarizer - Scikit uses FeatureUnion 

#FeatureUnion - You give it a list of transformers (which can be entire transformer pipelines) and when its transform() method is called then it runs each transform method in parallel
#waiting for their output and then concatenates and returns the result - this will be calling in the fit() method calls each transformer's fit() method

#a pipeline handling numerical and categorical attributes may look like this:

from sklearn.pipeline import FeatureUnion 

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy = "median")), 
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('one_hot_encoder', OneHotEncoder(sparse_output=False)),])

housing

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),])


#running the whole pipeline simply: 

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


#linear regression secion:

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:\t", lin_reg.predict(some_data_prepared))

print("labels:\t\t", list(some_labels))


#important to measure the RMSE

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse #having an error of $68k in housing predictions is not satisfactory. This error generally occurs when the model is underfitting the training data. When this happens this means that the features do not provide enough info to make good predictions or the model is not powerful enough 


#how to fix? we need amore powerful model to feed the training algorithm with better features/reduce contraints of the model 



#lets train DecisionTreeRegressor


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

#model is now trained - time to evaluate the training set 

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse  = np.sqrt(tree_mse)
tree_rmse #this model gives no error which means the data is overfitted 

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
scoring = "neg_mean_squared_error", cv=10)


rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

display_scores(rmse_scores)


#fine tuning hyperparameter values

#using GridSearchCV - all you need to do is to tell it which hyperparameters you want it to experiment with and what values to try out 

#for example the code below searches for he best combination of hyperparamter for the RandomForestRegressor 

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor


param_grid = [
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features':[2,3,4]},
]


forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring = 'neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_


grid_search.best_estimator  #30 is the max values of n_estimators that were evaluated. It's important to eval higher values. This could lead to the scoring improving


#a way to get the best estimator directly, 

grid_search.best_estimator_


# evaluation scores are also available 

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params) #the better solution is to set the hyperparameter to 6 and the n_estimator to 30. 

#rmse score for this 49k and it is slightly better than the 52k from earlier
    


#randomforestregressor can indicate the relative importance of each attribute for making accurate predictions


feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances

#importance scores nest to their corresponding attribute names: 

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_rooms"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


#time to evaluate system on the test set 

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_rmse #$48,126. 
#From this model we can see that the median income is the number one predictor of housing prices. 
