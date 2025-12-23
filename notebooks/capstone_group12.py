#!/usr/bin/env python
# coding: utf-8

# In[2]:
#function1: assign name to placeholder
def create_display_copy(df):
    sex_map = {1: "Male", 2: "Female"}
    insurance_map = {1: "Private", 2: "Public", 3: "Uninsured"}
    race_map = {1: "White", 2: "Black", 3: "Asian", 4: "Native", 5: "Other", 6: "Multiple"}
    region_map = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
    disease_map = {-8: "Don't Know", -7: "Refused", -1: "Inapplicable", 1: "Yes", 2: "No"}

    df_display = df.copy()

    # Assign new label columns (do not overwrite originals)
    if 'Sex' in df_display.columns:
        df_display['Sex_Label'] = df_display['Sex'].map(sex_map)
    if 'Insurance_Status' in df_display.columns:
        df_display['Insurance_Label'] = df_display['Insurance_Status'].map(insurance_map)
    if 'Race' in df_display.columns:
        df_display['Race_Label'] = df_display['Race'].map(race_map)
    if 'Region' in df_display.columns:
        df_display['Region_Label'] = df_display['Region'].map(region_map)

    for col in ['Heart_Disease', 'Stroke', 'Angina', 'Heart_Attack']:
        if col in df_display.columns:
            df_display[col + '_Label'] = df_display[col].map(disease_map)

    return df_display


# Function 2: Keep only rows with "Yes" or "No" (1 or 2) for disease variables
def filter_binary_disease_responses(df, disease_vars):
    return df[df[disease_vars].apply(lambda row: row.isin([1, 2]).all(), axis=1)].copy()


# Function 3: Train logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def run_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    return clf, X_test, y_test, report




# In[ ]:




