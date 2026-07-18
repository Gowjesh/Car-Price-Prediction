##PROBLEM STATEMENT:
#User Car Price Prediction

#Files
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from xgboost import XGBRegressor

##LOAD DATASET
df=pd.read_csv("./car_data.csv")

##EDA
print(df.head())
print(df.info())
print(df.describe())

##Handle Missing Values
df.dropna(inplace=True)

##Outlier Treatment
Q1=df['Price'].quantile(0.25)
Q3=df['Price'].quantile(0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
df=df[(df['Price']>=lower) & (df['Price']<=upper)]

##Feature Scaling
#It is only use for the Logistic Regression, KNN, SVM, Linear Regression, Ridge, Lasso, ElasticNet. For other models, it can't be used.

##Feature Engineering
df['car_age'] = 2026 - df['Year']
df.drop('Year',axis=1,inplace=True)

# ##Encoding
# le=LabelEncoder()
# df['Owner']=le.fit_transform(df['Owner'])
make_model = (
    df.groupby("Make")["Model"]
      .unique()
      .apply(list)
      .to_dict()
)

df=pd.get_dummies(df,columns=["Make",
        "Model",
        "Fuel_Type",
        "Transmission",
        "Location",
        "Color",
        "Owner",
        "Seller_Type",
        "Engine",
        "Max_Power",
        "Max_Torque",
        "Drivetrain"],
    drop_first=True)

##Train-Test Split
X=df.drop('Price',axis=1)
X.columns=(X.columns.astype(str).str.replace(r"[\[\]<>]","",regex=True))
y=df['Price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

###Feature Scaling
scaler=StandardScaler()
x_train=scaler.fit_transform(X_train)
x_test=scaler.transform(X_test)

##Models
lr=LinearRegression()
ridge=Ridge(alpha=0.1)
lasso=Lasso(alpha=0.1)
elastic=ElasticNet(alpha=0.1,l1_ratio=0.5)
rf=RandomForestRegressor(n_estimators=100,random_state=42)
xgb=XGBRegressor(n_estimators=50,learning_rate=0.1,random_state=42)

##Training
lr.fit(x_train,y_train)
ridge.fit(x_train,y_train)
lasso.fit(x_train,y_train)
elastic.fit(x_train,y_train)
rf.fit(X_train,y_train)
xgb.fit(X_train,y_train)

##Prediction
#By using this only,Because model comparison uses the pred of each model and then compare the accuracy of each model.For that we need to use the fit(trained data) and predict(test data) of each model and then compare the accuracy of each model.

lr_pred=lr.predict(x_test)
ridge_pred=ridge.predict(x_test)
lasso_pred=lasso.predict(x_test)
elastic_pred=elastic.predict(x_test)
rf_pred=rf.predict(X_test)
xgb_pred=xgb.predict(X_test)

##Evaluation Function
def evaluate(y_test,pred,n,p):
    mae=mean_absolute_error(y_test,pred)
    mse=mean_squared_error(y_test,pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test,pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return mae,mse,rmse,r2,adj_r2

##Model Comparison
models={
    "Linear Regression":lr_pred,
    "Ridge":ridge_pred,
    "Lasso":lasso_pred,
    "ElasticNet":elastic_pred,
    "Random Forest":rf_pred,
    "XGBoost":xgb_pred
}

result=[]
n=X_test.shape[0] #no of test samples
p=X_test.shape[1] #no of features
for name,pred in models.items():
    mae,mse,rmse,r2,adj_r2=evaluate(y_test,pred,n,p)

    result.append([name,mae,mse,rmse,r2,adj_r2])

result_df=pd.DataFrame(result,columns=["Model","MAE","MSE","RMSE","R2_Score","Adjusted_R2_Score"])

result_df=result_df.sort_values(by="R2_Score",ascending=False)

print(result_df)
print(X.columns.tolist())
#5-Fold Cross Validation Score
cv_score=cross_val_score(rf,X,y,cv=5,scoring="r2")
print("\nCross Validation Score:", cv_score.mean())

##Best Model Pridiction
best_model_name=result_df.iloc[0]['Model']
print("\nBest Model:",best_model_name)

##Model Saving
if best_model_name == "Linear Regression":
    best_model = lr
elif best_model_name == "Ridge Regression":
    best_model = ridge
elif best_model_name == "Lasso Regression":
    best_model = lasso
elif best_model_name == "Elastic Net":
    best_model = elastic
elif best_model_name == "Random Forest":
    best_model = rf
else:
    best_model = xgb

joblib.dump(best_model,'car_price.pkl')
joblib.dump(X.columns.tolist(),'columns.pkl')
joblib.dump(make_model, "make_model.pkl")
if best_model not in ["Random Forest","XGBoost"]:
    joblib.dump(scaler,'scaler.pkl')
print("\nSaved Data")
