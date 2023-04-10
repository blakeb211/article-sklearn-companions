# Title
Better Modeling with Sklearn Companion Libs
# Subtitle
Create uncluttered pipelines, validate faster, and utilize additional algorithms like AutoML from the comfort of your *Sklearn* skillset

# Motivation
I found a few Sklearn insights during a machine learning deep dive last year (see Sources). An informal poll of working data scientists revealed that these are not common knowledge. 

![Score Barplot](./figures/score_barplot.png)

# Preliminaries
* Insights are demonstrated on the Ames housing data set.
* Code snippets were taken from [this respository](https://github.com/blakeb211/ames-housing)
* The reader should have a little bit of experience with Sklearn to fully appreciate these. 
Let's get to it.

### Simplify your pipeline code with Feature Engine's **SklearnTransformWrapper**

After ingestion and creating a train-test split, we construct preprocessing recipes called **Pipelines** to automate our preprocessing steps such as scaling and one-hot encoding. These are easy to read and share, while also guarding against subtle data leakage during K-Fold validation. E.g. If a feature is scaled before creating the train-test split, this results in leakage.

Feature Engine's **SklearnTransformWrapper** wraps our 'StandardScaler()' and 'OneHotEncoder()' objects so that we can put them directly into our Pipeline object. No more ColumnTransformer. Data exits each step as a dataframe, so we don't need to fumble with numpy ndarrays, which are returned by default from most transformers.

The Sklearn helper method **make_pipeline*** creates our pipeline object in one line, naming each step for us. Now our *GridSearchCV* object can send lists of parameters to the pipeline using these names. Just query the step names with `pipe.named_steps`. For this example, it tells us that our estimator (the final step) is called 'elasticnet'. E.g. 'elasticnet__alpha' sends the list of values in *alpha_range* to the ElasticNet model at the end of our pipeline.

```
# skipping typical sklearn imports and dataframe creation 

from feature_engine.wrappers import SklearnTransformerWrapper

categoric_cols = X_train.select_dtypes(include=object).columns.tolist()

std_scaler = SklearnTransformerWrapper(transformer=StandardScaler())
OH_encoder = SklearnTransformerWrapper(transformer=OneHotEncoder(
    sparse_output=False, 
    drop='if_binary', 
    min_frequency=0.1, 
    handle_unknown='ignore'), 
    variables=categoric_cols)

pipe = make_pipeline(std_scaler, OH_encoder, ElasticNet(max_iter=2000))
alpha_range = np.linspace(70, 150, num=20)
gs = GridSearchCV(n_jobs=3, estimator=pipe, cv=10, scoring='neg_root_mean_squared_error', param_grid={
                  'elasticnet__l1_ratio': [0.7, 0.8, 0.9, 1.0], 'elasticnet__alpha': alpha_range})
```

### Fast and pretty model validation plots with Yellowbrick

The **Yellowbrick** companion library has helper classes and functions to automatically create various important graphics and keep our thinking up above the matplotlib api.

We can select one of the hyperparameters that we searched with *GridSearchCV* and validate it with Yellowbrick's **ValidationCurve** object. 

```
# Create validation curve to increase confidence that algorithm is performing reasonably
pipe_validation = make_pipeline(
    std_scaler, OH_encoder, ElasticNet(l1_ratio=1.0, max_iter=2000))

# Yellowbrick magic. Similar to GridSearchCV on a single parameter, but with a plot produced.
viz = ValidationCurve(
    pipe_validation, cv=10, param_name='elasticnet__alpha', param_range=alpha_range
)

# Fit and show the visualizer
viz.fit(X_train, y_train)
viz.show()
```

![Validation Curve](./figures/elasticnet_validation_curve.png)


### Add AutoML and Extreme Gradient Boosting to your algorithm toolbox with AutoSklearn and XGBoost
Sklearn has a range of estimators including regularized regression, support vector machine, single decision trees, bagged trees, boosted trees, and deep neural nets. We can add extreme gradient boosting and AutoML functionality by importing a single package for each.

**XGBoost** is a popular implementation of gradient boosted trees.

```
# ... sklearn imports, create onehot encoder transform, create dataframes ...
from xgboost import XGBRegressor

categoric_cols = X_train.select_dtypes(include=object).columns.tolist()
OH_encoder = SklearnTransformerWrapper(transformer=OneHotEncoder(
    sparse_output=False, drop='if_binary', min_frequency=0.1, handle_unknown='ignore'), variables=categoric_cols)

# Being a companion library, the XGBRegressor can be used inside a pipeline object just like a built in estimator
pipe = make_pipeline(OH_encoder, XGBRegressor(
    booster='gbtree', n_jobs=2, random_state=42))


# scan the number of trees and tree depth parameters for optimal values
num_trees_range = np.int64(np.linspace(200, 400.0, num=5))
depth_range = [1, 2, 3, 4]
gs = GridSearchCV(n_jobs=8, estimator=pipe, cv=10,
                  scoring='neg_root_mean_squared_error', param_grid={'xgbregressor__max_depth': depth_range,
                                                                     'xgbregressor__n_estimators': num_trees_range})
```

**Auto-sklearn** is a package that implements an 'automl' algorithm. Note that there are several different algorithms and really the term is an umbrella term for machine learning that requires very little user input. **AutoML Disclaimer** We do not want to give the impression it is a panacea, because it is not. In some instances using automl may simply shift your effort to the feature engineering phase. It is also quite possible that the complicated ensemble model produced cannot be put into production due to the difficulty implementing it on your tech stack. Lastly, if interpertability is an issue, for example in a regulated setting with outside auditors, you may prefer a simpler model for that reason.

 This is not a magical bullet.

```
# .. typical sklearn and other data science imports ... 
# Commonly occuring pthread errors with auto-sklearn are fixed by these two lines.
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'                    # DONT ASK 
from autosklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
import autosklearn.regression

# On the Ames dataset, the settings below dataset generate an ensemble 
# of LinearSVR and AdaBoost models that outperforms 
# our best manually-obtained model.

# By default, auto-sklearn trains for 1 hour and outputs the best model
# that it found.

# Below is a control block to save the model to a .pkl file so that 
# we do not have to re-run a lengthy training session over and over
# while we tweak our notebook.

automl = {}
filename_automl = "saved_model_automl.pkl"

X_train, X_test, y_train, y_test = make_train_test()

if os.path.exists(filename_automl):
    automl = joblib.load(filename=filename_automl)
else:
    categoricals = X_train.select_dtypes(object).columns.tolist()
    X_train[categoricals] = X_train.select_dtypes(
        object).astype('category')
    scorer = root_mean_squared_error

    automl = autosklearn.regression.AutoSklearnRegressor(metric=scorer)
    automl.fit(X_train, np.float64(y_train.to_numpy()))

    joblib.dump(automl, filename=filename_automl, compress=6)

y_hat = automl.predict(X_test)
rmse = mean_squared_error(y_test, y_hat, squared=False)
```

### Place ingestion and feature engineering code in an ingestion script  
Hopefully obvious by now, but place your code that creates ready-to-model pandas dataframes into functions inside of a script. A good name for this might be 'ingestion.py'. Call this script from all your notebooks and modeling scripts. Good names for the functions are 'make_frames' and 'make_cleaned'. *This way, ingestion or feature engineering code only needs to be changed in one place. Dependent code can simply be re-executed.* 

# Companion lib review
* [Feature-Engine](https://github.com/feature-engine/feature_engine) provides tranformers that can be easier to use than 
the Sklearn defaults, reducing boilerplate and tightening up pipelines
* [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) provides wrappers over estimators to make common 
visualizations quickly
* [xgboost](https://github.com/dmlc/xgboost) provides an additional (and widely popular) gradient boosting implementation
* [auto-sklearn](https://github.com/automl/auto-sklearn) provides an automl implementation

# Sources 
1. Georgetown Data Science Certificate Program
1. Hands-on Machine Learning with R book, Boehmke and Greenwell
1. Vectors Matrices and Least Squares book, Boyd and Vandenberghe
1. ThinkStats, Allen Downey
1. API docs and code for the various libraries Sklearn, Xgboost, AutoSklearn, Numpy, Pandas
