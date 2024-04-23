import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate 
from sklearn.metrics import confusion_matrix
import seaborn as sns

# load the data

breeds_data = pd.read_csv(r"C:\Users\林承劭\Desktop\大學課程\coding\python\Data Science\Final Project\breeds.csv")
breeds_data.index = breeds_data.index + 2


# find the missing value
missing_value = breeds_data.isnull().sum().sum()
print(f"Missing value: {missing_value}")

    # replaces all non-digit characters
breeds_data['weight'] = breeds_data['weight'].str.replace(r'\D', '', regex=True)
    #If any value in the 'weight' column cannot be converted to a numeric value, it will be replaced with NaN
breeds_data['weight'] = pd.to_numeric(breeds_data['weight'], errors='coerce')  
mean_weight = breeds_data['weight'].mean()
breeds_data['weight'] = breeds_data['weight'].fillna(mean_weight)


breeds_data['height'] = breeds_data['height'].str.replace(r'\D', '', regex=True)
breeds_data['height'] = pd.to_numeric(breeds_data['height'], errors='coerce')
mean_height = breeds_data['height'].mean()
breeds_data['height'] = breeds_data['height'].fillna(mean_height)

        # Check again if there is any missing value
print(breeds_data.isnull().sum())

Cbreeds_data = breeds_data.copy() # categorical 
Nbreeds_data = breeds_data.copy() # numerical

# find the type with object (categorical feature)
categorical_features = Cbreeds_data.select_dtypes(include='object').columns
print(categorical_features)


    # use label-Encoder to change the categorical feature into numerical feature
labelEncoder = LabelEncoder()
for category in ['breed','url','breed_group', 'life_span']:
    Cbreeds_data[category] = labelEncoder.fit_transform(Cbreeds_data[category])   


labelEncoder = LabelEncoder()
for category in ['breed_group', 'life_span']:
    Nbreeds_data[category] = labelEncoder.fit_transform(Nbreeds_data[category])
#print(Cbreed_data.info())

"""
# visualize the breed data after numerical 
for column in Nbreeds_data.columns[3:-4]: 
    Nbreeds_data[column].value_counts().plot(kind="bar")
    plt.title(column)  
    plt.xticks(ticks=range(5), labels=range(1, 6))
    plt.show()
"""


# Check if the dataset is balanced

apartment_living = Cbreeds_data.drop("a1_adapts_well_to_apartment_living", axis = 1).values
tendency = Cbreeds_data['d5_tendency_to_bark_or_howl'].values
tendency = tendency/ np.max(tendency)

print(f"\n#sample:{apartment_living.shape[0]} #feature: {apartment_living.shape[1]}")


grade_counts = Cbreeds_data['a1_adapts_well_to_apartment_living'].value_counts().sort_index()
#print(grade_counts)

# visualize 
plt.figure(figsize=(6,4))
plt.bar(grade_counts.index, grade_counts.values, color=['red', 'green', 'blue', 'yellow', 'purple'])
plt.xlabel('Rate')
plt.ylabel('Number of breeds')
plt.title('Distribution of Rate for Adapts well to apartment living')
plt.xticks(range(1, 6))
plt.show()

# Split the data into two subsets and normalize the features of samples

X = Cbreeds_data[['a_adaptability', 'a1_adapts_well_to_apartment_living', 'a2_good_for_novice_owners', 'a3_sensitivity_level', 'a4_tolerates_being_alone', 'a5_tolerates_cold_weather', 'a6_tolerates_hot_weather']]
y = Cbreeds_data['a1_adapts_well_to_apartment_living']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14040114613180515, random_state=42)

# Normalize features using the same scaler for both train and test sets
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)
print(f"#sample:{X_train.shape[0]} #feature: {X_test.shape[1]}")

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate the model on training set
y_train_pred = lr.predict(X_train)
mae_train = mean_absolute_error(y_train_pred, y_train)
mse_train = mean_squared_error(y_train_pred, y_train)
rmse_train = np.sqrt(mse_train)

print("\nLinear Regression Model Performance: ")
print('\nTraining set:')
print('MAE is: {}'.format(mae_train))
print('MSE is: {}'.format(mse_train))
print('RMSE is: {}'.format(rmse_train))


# Evaluate the model on testing set
y_test_pred = lr.predict(X_test)
mae_test = mean_absolute_error(y_test_pred, y_test)
mse_test = mean_squared_error(y_test_pred, y_test)
rmse_test = np.sqrt(mse_test)
print('\nTesting set:')
print('MAE is: {}'.format(mae_test))
print('MSE is: {}'.format(mse_test))
print('RMSE is: {}'.format(rmse_test))


# Labels for the features
labels = ["Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, y_test[:5], width, label='Ground Truth')  # Display the ground truth values for the first 5 samples
rects2 = ax.bar(x + width/2, y_test_pred[:5], width, label='Prediction')  # Display the prediction values for the first 5 samples

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Values')
ax.set_title('Ground Truth vs Prediction for the first 5 samples')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# visualize 
plt.show()


#ridge regression model to do prediction

x = np.random.rand(10, 1)
noise = np.random.rand(10, 1) * 0.2
y = 3 * x + noise

lr = Ridge(alpha=0.1)

lr.fit(X_train, y_train)
print("\ncoefficients is " + str(lr.coef_))

# Predict target variable using the Ridge model
y_test_pred_ridge = lr.predict(X_test)


# Ridge regression model performance
mae_ridge = mean_absolute_error(y_test_pred_ridge, y_test)
mse_ridge = mean_squared_error(y_test_pred_ridge, y_test)
rmse_ridge = np.sqrt(mse_ridge)

'''
# Compare with Linear Regression model performance
print('\nLinear Regression Model Performance:')
print('MAE is: {}'.format(mae_test))
print('MSE is: {}'.format(mse_test))
print('RMSE is: {}'.format(rmse_test))
'''

# tune the hyperparameters to get good performance

alpha_values = [0.1, 1.0, 5.0]
for alpha_value in alpha_values:
    # setting new alpha in the list
    ridge = Ridge(alpha=alpha_value) 
    ridge.fit(X_train, y_train)
    y_test_pred_ridge = ridge.predict(X_test)
    mae_ridge = mean_absolute_error(y_test_pred_ridge, y_test)
    mse_ridge = mean_squared_error(y_test_pred_ridge, y_test)
    rmse_ridge = np.sqrt(mse_ridge)
    print('\nRidge Regression Model with alpha={}:'.format(alpha_value))
    print('MAE: {}'.format(mae_ridge))
    print('MSE: {}'.format(mse_ridge))
    print('RMSE: {}'.format(rmse_ridge))


feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5", "Feature 6", "Feature 7"]
plt.figure(figsize=(10, 6))
plt.bar(feature_names, lr.coef_)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression Coefficients')
plt.xticks(rotation=45)
plt.show()


#logistic regression model

LR = LogisticRegression()
LR.fit(X_train, y_train)


y_train_pred = LR.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred, average='macro')
recall_train = recall_score(y_train, y_train_pred, average= 'macro')
F1_train = f1_score(y_train, y_train_pred, average= "macro")

y_test_pred = LR.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred, average= "macro")
recall_test = recall_score(y_test, y_test_pred, average= "macro")
F1_test = f1_score(y_test, y_test_pred, average="macro")

print("\nLogistic Regression")
print('Training set:')
print('Accuracy: {}'.format(accuracy_train))
print('Precision: {}'.format(precision_train))
print('Recall: {}'.format(recall_train))
print('F-1 Score: {}'.format(F1_train))

print('\nTesting set:')
print('Accuracy: {}'.format(accuracy_test))
print('Precision: {}'.format(precision_test))
print('Recall: {}'.format(recall_test))
print('F-1 Score: {}'.format(F1_test))


cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()


# set the user interface 
headers = ["Code Option","Attribute", "Description"]

all_attributes = [
    ['a','a_adaptability', 'Adaptability'],
    ['a1','a1_adapts_well_to_apartment_living', 'Adapts well to apartment living'],
    ['a2','a2_good_for_novice_owners', 'Good for novice owners'],
    ['a3','a3_sensitivity_level', 'Sensitivity level'],
    ['a4','a4_tolerates_being_alone', 'Tolerates being alone'],
    ['a5','a5_tolerates_cold_weather', 'Tolerates cold weather'],
    ['a6','a6_tolerates_hot_weather', 'Tolerates hot weather'],
    ['b','b_all_around_friendliness', 'All around friendliness'],
    ['b1','b1_affectionate_with_family', 'Affectionate with family'],
    ['b2','b2_incredibly_kid_friendly_dogs', 'Incredibly kid friendly dogs'],
    ['b3','b3_dog_friendly', 'Dog friendly'],
    ['b4','b4_friendly_toward_strangers', 'Friendly toward strangers'],
    ['c','c_health_grooming', 'Health grooming'],
    ['c1','c1_amount_of_shedding', 'Amount of shedding'],
    ['c2','c2_drooling_potential', 'Drooling potential'],
    ['c3','c3_easy_to_groom', 'Easy to groom'],
    ['c4','c4_general_health', 'General health'],
    ['c5','c5_potential_for_weight_gain', 'Potential for weight gain'],
    ['c6','c6_size', 'Size'],
    ['d','d_trainability', 'Trainability'],
    ['d1','d1_easy_to_train', 'Easy to train'],
    ['d2','d2_intelligence', 'Intelligence'],
    ['d3','d3_potential_for_mouthiness', 'Potential for mouthiness'],
    ['d4','d4_prey_drive', 'Prey drive'],
    ['d5','d5_tendency_to_bark_or_howl', 'Tendency to bark or howl'],
    ['d6','d6_wanderlust_potential', 'Wanderlust potential'],
    ['e','e_exercise_needs', 'Exercise needs'],
    ['e1','e1_energy_level', 'Energy level'],
    ['e2','e2_intensity', 'Intensity'],
    ['e3','e3_exercise_needs', 'Exercise needs'],
    ['e4','e4_potential_for_playfulness', 'Potential for playfulness']
]

print(tabulate(all_attributes, headers = headers, tablefmt='grid'))

# Let the user pick at least 2 two breed preferences

breeds_attributes = {}
min_attributes = 2  
while True:
    code = input("Please pick your preference above (if finished type 'done'): ")
    if code.lower() == 'done':
        if len(breeds_attributes) < min_attributes:
            print(f"Please select {min_attributes - len(breeds_attributes)} more attributes.")
            continue
        else:
            break
    elif code.lower() not in [attr[0].lower() for attr in all_attributes]:
        print("Please enter a valid attribute code.")
        continue

    for attr in all_attributes:
        if attr[0].lower() == code.lower():
            if attr[1] not in breeds_attributes:
                breeds_attributes[attr[1]] = None  # Initialize the attribute with None
            else:
                print("This attribute has already been selected. Please pick another one.")

print('\nSelected Attributes:', breeds_attributes)

#　let the user to rate each feature for a dog breed on a scale of 1 to 5
user_dog_features = []
for attribute in breeds_attributes:
    while True:
        feature_value = int(input(f"\nPlease enter the value for [{attribute}] (a number from 1 to 5): "))
        if 1 <= feature_value <= 5:
            user_dog_features.append(feature_value)
            break
        else:
            print("Please enter a valid number (1 to 5).")
            pass

similarities = []
for index, row in breeds_data.iterrows():
    dog_features = []
    for attribute in breeds_attributes:
        dog_features.append(row[attribute])
    similarity = np.corrcoef(user_dog_features, dog_features)[0, 1]
    similarities.append(similarity)

breeds_data['similarity'] = similarities

# Calculate similarities and get top 10 similar dog breeds

pd.set_option('display.max_colwidth', 100) # expand the width of each column 

top_10_dogs = breeds_data.nlargest(10, 'similarity')[['breed', 'url', 'similarity']]

print('\n'.join(['{:^100}'.format('Top 10 Similar Dog Breeds')]))
print('\n')
print(top_10_dogs.to_string(index=False))
