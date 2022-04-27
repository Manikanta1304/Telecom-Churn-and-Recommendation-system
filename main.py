import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "./Telco_customer_churn_services_Final_datasheet (1).xlsx"
df = pd.read_excel(DATA_PATH)

df_content = df[['Customer ID', 'Offer', 'Internet Type', 'Payment Method']]
df_content['Offer'] = df_content['Offer'].replace('None', 'Offer F')
df_content.set_index('Customer ID', inplace=True)
df_content['bow'] = df_content['Offer'] + ' ' + df_content['Internet Type'] + ' ' + df_content['Payment Method']
# df_content

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df_content['bow'])

# creating a Series for the customer IDs so they are associated to an ordered numerical
# list I will use later to match the indexes
indices = pd.Series(df_content.index)
cosine_sim = cosine_similarity(count_matrix, count_matrix)

df_proc = pd.read_csv('./Telco_customer_churn_services_Final_datasheet (1)_processed.csv')
# split df_proc in feature matrix and target vector
X = df_proc.drop('Churn', axis=1)
y = df_proc['Churn']

# split df_proc between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

# balance = pd.read_csv('./Telco_customer_churn_services_Final_datasheet (1)_processed_balaced.csv')
#
# X_test['Customer ID'] = balance['Customer ID']
X_test.reset_index(inplace=True, drop=True)
# print(X_test.isna().sum())

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.loc[:, X_train.columns != 'Customer ID'])
X_test1 = scaler.transform(X_test.loc[:, X_test.columns != 'Customer ID'])
X_test1 = pd.DataFrame(X_test1, columns=X_test.columns[0:33])
# print(X_test1.isna().sum())

def recommendations(title, cosine_sim):
    recommended_offers = []

    # getting the index of the customer ID that
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 3 most similar offers
    top_10_indexes = list(score_series.iloc[1:4].index)
    # print(top_10_indexes)

    # populating the list with the offers of the best 10 matching customers
    for i in top_10_indexes:
        recommended_offers.append(list(df_content['Offer'])[i])

    return recommended_offers


def predict(index):
    model_svm = pickle.load(open('./model_svm.sav', 'rb'))
    ypred = model_svm.predict(X_test1.loc[[index]])
    # print(ypred)

    if ypred[0] == 0:
        print('Customer {} is not likely to churn. Thank you!'.format(X_test.loc[[index]]['Customer ID'].values[0]))
    else:
        # print(X_test.loc[[index]]['Customer ID'].values[0])
        print('Customer is likely to churn and below are the list of offer suggestions to the {} customer\n'.format(X_test.loc[[index]]['Customer ID'].values[0]),
              recommendations(X_test.loc[[index]]['Customer ID'].values[0], cosine_sim=cosine_sim))


if __name__ == "__main__":
    print('Below sample dataframe to enter the index\n', X_test.head(10))
    value = int(input('Enter the index of the customer ID to predict the offers: '))
    predict(value)