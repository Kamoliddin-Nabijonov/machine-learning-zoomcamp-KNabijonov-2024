{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "3190ad6b-d39d-40fc-91fd-df634f684173",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    " \n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import KFold\n",
    " \n",
    "from sklearn.feature_extraction import DictVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import roc_auc_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f596e20e-f3e6-4147-b415-7ebf1ffa5610",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data preparation\n",
    "df = pd.read_csv('churn.csv')\n",
    " \n",
    "df.columns = df.columns.str.lower().str.replace(' ', '_')\n",
    " \n",
    "categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)\n",
    " \n",
    "for c in categorical_columns:\n",
    "    df[c] = df[c].str.lower().str.replace(' ', '_')\n",
    " \n",
    "df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')\n",
    "df.totalcharges = df.totalcharges.fillna(0)\n",
    " \n",
    "df.churn = (df.churn == 'yes').astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "4f7e060b-fbe2-432a-8bc8-07b29a64ebbe",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data splitting\n",
    "df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "494c5426-636b-45fa-8c74-fcb827ba221a",
   "metadata": {},
   "outputs": [],
   "source": [
    "numerical = ['tenure', 'monthlycharges', 'totalcharges']\n",
    " \n",
    "categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',\n",
    "       'phoneservice', 'multiplelines', 'internetservice',\n",
    "       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',\n",
    "       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',\n",
    "       'paymentmethod']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "faf962fc-984f-4b78-8a04-1e5f7a6ee37e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train(df_train, y_train, C=1.0):\n",
    "    dicts = df_train[categorical + numerical].to_dict(orient='records')\n",
    " \n",
    "    dv = DictVectorizer(sparse=False)\n",
    "    X_train = dv.fit_transform(dicts)\n",
    " \n",
    "    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)\n",
    "    model.fit(X_train, y_train)\n",
    " \n",
    "    return dv, model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "7751dec8-a394-49e4-bd6f-f6bd7cd64407",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(df, dv, model):\n",
    "     dicts = df[categorical + numerical].to_dict(orient='records')\n",
    " \n",
    "     X = dv.transform(dicts)\n",
    "     y_pred = model.predict_proba(X)[:,1]\n",
    " \n",
    "     return y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "1b085904-33e3-4d60-a02f-13a94e7c7ec4",
   "metadata": {},
   "outputs": [],
   "source": [
    "C = 1.0\n",
    "n_splits = 5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "b72dbfe3-a08b-49d3-811d-067856060eb9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C=1.0 0.841 +- 0.007\n"
     ]
    }
   ],
   "source": [
    "kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)  \n",
    " \n",
    "scores = []\n",
    " \n",
    "for train_idx, val_idx in kfold.split(df_full_train):\n",
    "    df_train = df_full_train.iloc[train_idx]\n",
    "    df_val = df_full_train.iloc[val_idx]\n",
    " \n",
    "    y_train = df_train.churn.values\n",
    "    y_val = df_val.churn.values\n",
    " \n",
    "    dv, model = train(df_train, y_train, C=C)\n",
    "    y_pred = predict(df_val, dv, model)\n",
    " \n",
    "    auc = roc_auc_score(y_val, y_pred)\n",
    "    scores.append(auc)\n",
    " \n",
    "print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))\n",
    " \n",
    "# Output: C=1.0 0.841 +- 0.008"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "29622556-538a-4146-bc96-3519d4913e95",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8579400803839363"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)\n",
    "y_pred = predict(df_test, dv, model)\n",
    "y_test = df_test.churn.values\n",
    " \n",
    "auc = roc_auc_score(y_test, y_pred)\n",
    "auc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "2efa4e96-a41b-41c5-a252-03ceb6ae5b2a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "f61a8e86-5786-429e-a9cd-7f39b5fe996b",
   "metadata": {},
   "outputs": [],
   "source": [
    "output_file = f'model_C={C}.bin'\n",
    "\n",
    "with open(output_file, 'wb') as f_out:\n",
    "    pickle.dump((dv, model), f_out)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "75f63f59-2b3c-4bca-afbd-dbb4d60e59a7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ee30d141-60c1-473b-93e9-e220aeef851a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "model_file = 'model_C=1.0.bin'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9529ced9-19e9-4c11-b29c-86d894b70e96",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(DictVectorizer(sparse=False),\n",
       " LogisticRegression(max_iter=1000, solver='liblinear'))"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "with open(model_file, 'rb') as f_in:\n",
    "    dv, model = pickle.load(f_in)\n",
    " \n",
    "dv, model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5c00b3d3-9ab1-4bf8-aed9-bfd31263fa89",
   "metadata": {},
   "outputs": [],
   "source": [
    "customer = {\n",
    "    'gender': 'female',\n",
    "    'seniorcitizen': 0,\n",
    "    'partner': 'yes',\n",
    "    'dependents': 'no',\n",
    "    'phoneservice': 'no',\n",
    "    'multiplelines': 'no_phone_service',\n",
    "    'internetservice': 'dsl',\n",
    "    'onlinesecurity': 'no',\n",
    "    'onlinebackup': 'yes',\n",
    "    'deviceprotection': 'no',\n",
    "    'techsupport': 'no',\n",
    "    'streamingtv': 'no',\n",
    "    'streamingmovies': 'no',\n",
    "    'contract': 'month-to-month',\n",
    "    'paperlessbilling': 'yes',\n",
    "    'paymentmethod': 'electronic_check',\n",
    "    'tenure': 1,\n",
    "    'monthlycharges': 29.85,\n",
    "    'totalcharges': 29.85\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "772b73b7-35f3-4348-bf9c-1384e3d0477f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1.  ,  0.  ,  0.  ,  1.  ,  0.  ,  1.  ,  0.  ,  0.  ,  1.  ,\n",
       "         0.  ,  1.  ,  0.  ,  0.  , 29.85,  0.  ,  1.  ,  0.  ,  0.  ,\n",
       "         0.  ,  1.  ,  1.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  1.  ,\n",
       "         0.  ,  0.  ,  1.  ,  0.  ,  1.  ,  0.  ,  0.  ,  1.  ,  0.  ,\n",
       "         0.  ,  1.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  1.  , 29.85]])"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = dv.transform([customer])\n",
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "15ebce7e-c109-4e46-9160-21435ec7c408",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6433012217921572"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict_proba(X)[0,1]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "ml-zoomcamp",
   "language": "python",
   "name": "ml-zoomcamp"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
