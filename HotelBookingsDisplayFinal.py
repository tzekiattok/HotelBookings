import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np
import pandas as pd

# Import tools needed for visualization
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.tree import export_graphviz
import seaborn as sns
import matplotlib.pyplot as plt

# Models & Processing
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn import metrics

# Evaluation Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score

# Tuning of Model
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV

# Imblearn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

st.set_page_config(layout="wide")
st.write("""
# Predicting the probability of Hotel Bookings being cancelled
""")

st.sidebar.header('User Input Parameters')

# Fill missing values in children with 0  
# Assuming that the 4 NaN values means that there is no children
df = pd.read_csv('https://github.com/tzekiattok/HotelBookings/blob/main/hotel_bookings.xls?raw=true')
df['children'] = df['children'].fillna(0)

# data encoding
# Split date into year, month
df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])
df["reservation_status_year"] = df["reservation_status_date"].dt.year
df["reservation_status_month"] = df["reservation_status_date"].dt.month
# Check if booking was made through agent
df["through_agent"] = df["agent"].notnull().map({True: 1, False: 0})


# Choose columns we want to include in dataset
# Removed some columns that will not be useful:

numerical_features = [
    #"lead_time", 
    "total_of_special_requests", #"required_car_parking_spaces",
    "booking_changes"
    , "previous_cancellations", "adults",
    "previous_bookings_not_canceled",
]

categorical_features = [
    "hotel","meal", "market_segment", "distribution_channel",
    "is_repeated_guest", "reserved_room_type", "deposit_type", "customer_type",
    "reservation_status_month", "through_agent",
]


# Separate cat and num features

# [:] creates a copy of the dataframe instead of a view on the original dataframe.
# This prevents mutations on X_cat and X_num from affecting df.
X_cat = df[categorical_features][:]
X_num = df[numerical_features][:]

def with_dummies(df, col_name):
    return pd.concat([
        df,
        pd.get_dummies(df[col_name], prefix=col_name, prefix_sep='=', drop_first=True),
    ], axis=1).drop([col_name], axis=1)

# iteratively replace all categorical columns with one-hot encodings

for feature in categorical_features:
    X_cat = with_dummies(X_cat, feature)
    
X = pd.concat([X_cat, X_num], axis=1)
y = df["is_canceled"]
#Drop distribution channel due to  multicolinearity
X = X.drop(['distribution_channel=Direct',
            'distribution_channel=GDS',
            'distribution_channel=TA/TO',
            'distribution_channel=Undefined'], axis=1)
st.subheader('Independent Variables')
st.write(X)
st.subheader('Dependent Variable')
st.write(y)

def user_input_features():
    hotel = st.sidebar.selectbox('Hotel', ('City Hotel','Resort Hotel'))
    #is_canceled = st.sidebar.slider('is_canceled', 0,1)
    #lead_time =st.sidebar.slider('lead_time',1,500)
    
    #arrival_date_year = st.sidebar.slider('arrival_date_year',2015,2017)
    #arrival_date_month =  st.sidebar.slider('arrival_date_month',1,12)
    #arrival_date_week_number = st.sidebar.slider('arrival_date_week_number',1,53)
    #arrival_date_day_of_month = st.sidebar.slider('arrival_date_day_of_month',1,12)
    total_of_special_requests = st.sidebar.slider('total_of_special_requests',0,5)
    #required_car_parking_spaces = st.sidebar.slider('required_car_parking_spaces',0,8)
    booking_changes= st.sidebar.slider('booking_changes',0,21)
    #reservation_status_year = st.sidebar.slider('reservation_status_year',2015,2017)
    previous_cancellations = st.sidebar.slider('previous_cancellations',0,26)
    through_agent = st.sidebar.slider('through_agent ',0,1)
    is_repeated_guest = st.sidebar.slider('is_repeated_guest',0,1)
    reservation_status_month =st.sidebar.slider('reservation_status_month',1,12)
    adults =st.sidebar.slider('adults',0,55)
    previous_bookings_not_canceled =st.sidebar.slider('previous_bookings_not_canceled',0,72)
    #days_in_waiting_list =st.sidebar.slider('days_in_waiting_list',0,365)
    #adr = st.sidebar.number_input('adr')
    #babies = st.sidebar.slider('babies',0,10)
    #stays_in_week_nights =st.sidebar.slider('stays_in_week_nights',0,50)
    #company =st.sidebar.number_input('company')
    #children =st.sidebar.number_input('children')
    #stays_in_weekend_nights = st.sidebar.slider('booking_changes',0,20)
    #Additional variables
    meal  = st.sidebar.selectbox('meal', ('FB','HB','SC','Undefined'))
    market_segment =  st.sidebar.selectbox('market_segment', ('Direct' ,'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups',
 'Undefined'))
    reserved_room_type = st.sidebar.selectbox('reserved_room_type', ('B','C',  'D', 'E', 'G', 'F', 'H', 'L', 'P'))
    deposit_type = st.sidebar.selectbox('deposit_type', ('Refundable', 'Non Refund'))
    customer_type=  st.sidebar.selectbox('customer_type', ('Transient',  'Transient-Party', 'Group'))
    
    
    
    
    
    data = {'hotel':hotel,
            #'lead_time':lead_time,
            #'arrival_date_year': arrival_date_year,
            #'arrival_date_month':arrival_date_month,
            #'arrival_date_week_number':arrival_date_week_number,
            #'arrival_date_day_of_month':arrival_date_day_of_month,
            'total_of_special_requests' :total_of_special_requests,
            #'required_car_parking_spaces' : required_car_parking_spaces,
            'booking_changes': booking_changes,
            'previous_cancellations' : previous_cancellations,
            'through_agent' : through_agent,
            'is_repeated_guest' : is_repeated_guest,
            'reservation_status_month' :reservation_status_month,
            'adults' : adults,
            'previous_bookings_not_canceled' :previous_bookings_not_canceled,
            #'days_in_waiting_list' :days_in_waiting_list,
            #'adr' : adr,
            #'babies' : babies,
            #'stays_in_week_nights' :stays_in_week_nights,
            #'company' :company,
            #'children' :children,
            #'stays_in_weekend_nights' : stays_in_weekend_nights,
            'meal'  : meal,
            'market_segment' : market_segment,
            'reserved_room_type' :reserved_room_type,
            'deposit_type' :deposit_type,
            'customer_type' : customer_type,
            }
    
#Encode categorical data
    finalData = {
    }
    finalData['hotel=Resort Hotel']=0
    if data['hotel']=='Resort Hotel':
       finalData['hotel=Resort Hotel']=1
    finalData['meal=FB']=0
    finalData['meal=HB']=0
    finalData['meal=SC']=0
    finalData['meal=Undefined']=0
    #finalData['meal=BB']=0
    if data['meal']=='FB':
        finalData['meal=FB']=1
    if data['meal']=='HB':
        finalData['meal=HB']=1
    if data['meal']=='SC':
        finalData['meal=SC']=1
    #if data['meal']=='BB':
    #    finalData['meal=BB']=1
    if data['meal']=='Undefined':
        finalData['meal=Undefined']=1
        
    #Market_Segment
    finalData['market_segment=Complementary']=0
    finalData['market_segment=Corporate']=0
    finalData['market_segment=Direct']=0
    finalData['market_segment=Groups']=0
    finalData['market_segment=Offline TA/TO']=0
    finalData['market_segment=Online TA']=0
    finalData['market_segment=Undefined']=0
    if data['market_segment']=='Complementary':
        finalData['market_segment=Complementary']=1
    if data['market_segment']=='Corporate':
        finalData['market_segment=Corporate']=1
    if data['market_segment']=='Direct':
        finalData['market_segment=Direct']=1
    if data['market_segment']=='Groups':
        finalData['market_segment=Groups']=1
    if data['market_segment']=='Offline TA/TO':
        finalData['market_segment=Offline TA/TO']=1
    if data['market_segment']=='Online TA':
        finalData['market_segment=Online TA']=1
    if data['market_segment']=='Undefined':
        finalData['market_segment=Undefined']=1
        
    #repeated guest
    finalData['is_repeated_guest=1']=0
    if data['is_repeated_guest'] ==1:
        finalData['is_repeated_guest=1']=1

    #ReservedRoom
    #finalData['reserved_room_type=A']=0
    finalData['reserved_room_type=B']=0
    finalData['reserved_room_type=C']=0
    finalData['reserved_room_type=D']=0
    finalData['reserved_room_type=E']=0
    finalData['reserved_room_type=F']=0
    finalData['reserved_room_type=G']=0
    finalData['reserved_room_type=H']=0
    finalData['reserved_room_type=L']=0
    finalData['reserved_room_type=P']=0
    #if data['reserved_room_type']=='A':
    #    finalData['reserved_room_type=A']=1
    if data['reserved_room_type']=='B':
        finalData['reserved_room_type=B']=1
    if data['reserved_room_type']=='C':
        finalData['reserved_room_type=C']=1
    if data['reserved_room_type']=='D':
        finalData['reserved_room_type=D']=1
    if data['reserved_room_type']=='E':
        finalData['reserved_room_type=E']=1
    if data['reserved_room_type']=='F':
        finalData['reserved_room_type=F']=1
    if data['reserved_room_type']=='G':
        finalData['reserved_room_type=G']=1
    if data['reserved_room_type']=='H':
        finalData['reserved_room_type=H']=1
    if data['reserved_room_type']=='L':
        finalData['reserved_room_type=L']=1
    if data['reserved_room_type']=='P':
        finalData['reserved_room_type=P']=1
        
    #deposit type
    #finalData['deposit_type=No Deposit']=0
    finalData['deposit_type=Non Refund']=0
    finalData['deposit_type=Refundable']=0
    #if data['deposit_type']=='No Deposit':
    #    finalData['deposit_type=No Deposit']=1
    if data['deposit_type']=='Refundable':
        finalData['deposit_type=Refundable']=1
    if data['deposit_type']=='Non Refund':
        finalData['deposit_type=Non Refund']=1
        
    #CustomerType
    finalData['customer_type=Group']=0
    finalData['customer_type=Transient']=0
    finalData['customer_type=Transient-Party']=0
    #finalData['customer_type=Contract']=0
    if data['customer_type']=='Transient':
        finalData['customer_type=Transient']=1
    elif  data['customer_type']=='Transient-Party':
        finalData['customer_type=Transient-Party']=1
    #elif data['customer_type']=='Contract':
    #    finalData['customer_type=Contract']=1
    else:
        finalData['customer_type=Group']=1
        
    
    
    #ReservationStatus
    finalData['reservation_status_month=2']=0
    finalData['reservation_status_month=3']=0
    finalData['reservation_status_month=4']=0
    finalData['reservation_status_month=5']=0
    finalData['reservation_status_month=6']=0
    finalData['reservation_status_month=7']=0
    finalData['reservation_status_month=8']=0
    finalData['reservation_status_month=9']=0
    finalData['reservation_status_month=10']=0
    finalData['reservation_status_month=11']=0
    finalData['reservation_status_month=12']=0
    if data['reservation_status_month']==2:
        finalData['reservation_status_month=2']=1
    if data['reservation_status_month']==3:
        finalData['reservation_status_month=3']=1
    if data['reservation_status_month']==4:
        finalData['reservation_status_month=4']=1
    if data['reservation_status_month']==5:
        finalData['reservation_status_month=5']=1
    if data['reservation_status_month']==6:
        finalData['reservation_status_month=6']=1
    if data['reservation_status_month']==7:
        finalData['reservation_status_month=7']=1
    if data['reservation_status_month']==8:
        finalData['reservation_status_month=8']=1
    if data['reservation_status_month']==9:
        finalData['reservation_status_month=9']=1
    if data['reservation_status_month']==10:
        finalData['reservation_status_month=10']=1
    if data['reservation_status_month']==11:
        finalData['reservation_status_month=11']=1
    if data['reservation_status_month']==12:
        finalData['reservation_status_month=12']=1
    #Hotel
        
    
      
    #through_agent
    finalData['through_agent=1']=0
    if data['through_agent'] ==1:
        finalData['through_agent=1']=1
   
    #finalData['lead_time']=lead_time
    
    #'arrival_date_year': arrival_date_year,
    #'arrival_date_month':arrival_date_month,
    #'arrival_date_week_number':arrival_date_week_number,
    #'arrival_date_day_of_month':arrival_date_day_of_month,
    finalData['total_of_special_requests'] = total_of_special_requests
    #finalData['required_car_parking_spaces'] = required_car_parking_spaces
    finalData['booking_changes'] = booking_changes
    finalData['previous_cancellations'] = previous_cancellations
    #'through_agent' : through_agent
    #'is_repeated_guest' : is_repeated_guest
    #'reservation_status_month' :reservation_status_month,
    finalData['adults'] = adults
    finalData['previous_bookings_not_canceled'] =previous_bookings_not_canceled
    #'days_in_waiting_list' :days_in_waiting_list,
    #'adr' : adr,
    #'babies' : babies,
    #'stays_in_week_nights' :stays_in_week_nights,
    #'company' :company,
    #'children' :children,
    #'stays_in_weekend_nights' : stays_in_weekend_nights,
    #'meal'  : meal,
    #'market_segment' : market_segment,
    #'reserved_room_type' :reserved_room_type,
    #'deposit_type' :deposit_type,
#'customer_type' : customer_type
    features = pd.DataFrame(finalData, index=[0])
    return features
    
    
        
    

dfuser = user_input_features()
print(dfuser)
st.subheader('User Input parameters')
st.dataframe(dfuser)


#############
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split

random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state=random_seed)
X_train_unscaled = X_train
X_test_unscaled = X_test
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # fit sc to X_train and transform X_train
X_test = sc.transform(X_test)       # transform X_test for final evaluation.

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)


###########Evaluation
from sklearn import metrics

def evaluate(y_pred, y_test):
    print(metrics.classification_report(y_test, y_pred))

    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    roc = metrics.roc_auc_score(y_test, y_pred)
    matthews = metrics.matthews_corrcoef(y_test, y_pred)
    print("Accuracy  :",  accuracy)
    print("Precision :",  precision)
    print("Recall    :",  recall)
    print("F1        :",  f1)
    print("ROC       :",  roc)
    print("Matthews  :",  matthews)
col1,col2,col3,col4 = st.columns(4)
###### Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dtreegs = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=5,max_depth=50)

#depth = [20, 30, 40, 50]
#depth = [5, 10, 20, 30, 40, 50]
#min_samples_leaf = [2, 3, 5, 8]
#param_grid = {
#    'max_depth': depth,
#    'min_samples_leaf': min_samples_leaf
#}

#gs = GridSearchCV(dtreegs, param_grid, cv=10, verbose=True, n_jobs=-1)
#gs.fit(X_train, y_train)
dtreegs.fit(X_train, y_train)
#dtreegs = gs.best_estimator_
#print(gs.best_params_)

nodes = dtreegs.tree_.node_count
train_score = dtreegs.score(X_train, y_train)
test_score = dtreegs.score(X_test, y_test)
print('Node count     :', nodes)
print('Training score :', train_score)
print('Testing score  :', test_score)
print('------------------------------------')
y_pred = dtreegs.predict(X_test)
evaluate(y_pred, y_test)
prediction = dtreegs.predict(dfuser)
col1.subheader('Decision Tree Classifier')
col1.write(dtreegs.predict_proba(dfuser))
col1.subheader('Results')
if prediction == 0:
    col1.write('Booking Not cancelled')
else:
    col1.write('Booking cancelled')
with col1:
    st.subheader('Algorithm Scores')
    st.write('Training score :', train_score)
    st.write('Testing score  :', test_score)

#############Logistic regression
from sklearn.linear_model import LogisticRegression

lrgs = LogisticRegression(C=0.1,
                          fit_intercept=False,
                          solver='lbfgs',
                          max_iter=400)
lrgs.fit(X_train, y_train)
train_score = lrgs.score(X_train, y_train)
test_score = lrgs.score(X_test, y_test)
print('Training score :', train_score)
print('Testing score  :', test_score)
print('------------------------------------')
y_pred = lrgs.predict(X_test)
evaluate(y_pred, y_test)
prediction = lrgs.predict(dfuser)
col2.subheader('Logistic Regression')
col2.write(lrgs.predict_proba(dfuser))
col2.subheader('Results')
if prediction == 0:
    col2.write('Booking Not cancelled')
else:
    col2.write('Booking cancelled')
with col2:
    st.subheader('Algorithm Scores')
    st.write('Training score :', train_score)
    st.write('Testing score  :', test_score)


########## Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rforestgs = RandomForestClassifier(n_estimators=100,max_depth= 13,max_features= 'sqrt',min_samples_leaf= 2,min_samples_split= 2)

rforestgs.fit(X_train, y_train)
train_score = rforestgs.score(X_train, y_train)
test_score = rforestgs.score(X_test, y_test)
print('Training score :', train_score)
print('Testing score  :', test_score)
print('------------------------------------')
y_pred = rforestgs.predict(X_test)
evaluate(y_pred, y_test)
prediction = rforestgs.predict(dfuser)
col3.subheader('Random Forest Classifier')
col3.write(rforestgs.predict_proba(dfuser))
col3.subheader('Results')
if prediction == 0:
    col3.write('Booking Not cancelled')
else:
    col3.write('Booking cancelled')
with col3:
    st.subheader('Algorithm Scores')
    st.write('Training score :', train_score)
    st.write('Testing score  :', test_score)
    
#ANN
import tensorflow as tf
from tensorflow import keras
batch_size = 32
epochs = 10
learning_rate = 0.01

inputs = keras.Input(shape=(44,))
x = keras.layers.Dense(64, activation=tf.nn.relu)(inputs)
x = keras.layers.Dense(64, activation=tf.nn.relu)(x)
outputs = keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

model_1 = keras.Model(inputs, outputs)
model_1.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                metrics=['accuracy'])

model_1.fit(X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=(X_test, y_test))
score = model_1.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print('Test loss:     ', score[0])
print('Test accuracy: ', score[1])
y_score = model_1.predict(X_test,batch_size=32, verbose=0)

threshold = 0.5

# convert score to actual predictions
y_pred = np.where(y_score > threshold, 1, 0)
evaluate(y_pred, y_test)
rounded_pred = np.argmax(y_pred,axis=-1)
#col4.write(rounded_pred)
col4.subheader('Probability of Neural Network; Threshhold for cancelled >0.5')
nn_pred = model_1.predict(dfuser,batch_size=32, verbose=0)
col4.write(nn_pred)
col4.subheader('Neural Network Results:')
nn_result ='Booking Cancelled'
if nn_pred<0.5:
    nn_result = 'Booking Not Cancelled'
    
col4.write(nn_result)
