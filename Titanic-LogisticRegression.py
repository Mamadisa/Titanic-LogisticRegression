import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pd.options.display.max_columns = None

# DATA INSPECTION:

titanic_data = pd.read_csv(
    r"C:\Users\bmama\Desktop\Data Sciences\DataScience & Machine Learning\13-Logistic-Regression\titanic_train.csv")
titanic_data.head(3)
titanic_data.info()
titanic_data.columns
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex',
# 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

# EXPLORATORY DATA ANALYSIS:

sns.heatmap(titanic_data.isnull(), yticklabels=False,
            cbar=False, cmap="coolwarm")
# Missing AGE, CABIN and one EMBARK
# We can drop CABIN

sns.set_style('whitegrid')
sns.countplot(x='Survived', data=titanic_data)

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex',
              data=titanic_data, palette='pastel')

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass',
              data=titanic_data)

sns.displot(titanic_data['Age'].dropna(), kde=False, bins=30)
# The older you get the less representation you have onboard

# Number of Siblings/Spouses
sns.countplot(x='SibSp', data=titanic_data)
# Alot of Single people on Board with no Children are Men
sns.set_style('whitegrid')
sns.countplot(x='SibSp', hue='Sex',
              data=titanic_data)

# How much people paid as a distribution
titanic_data['Fare'].hist(bins=40, figsize=(10, 4))

# Distrubutions is skewed to the Left(cheaper fare) which would indicate
# that most passangers are in the cheaper 3rd class.

# Relationship between 'Fare' 'Age' and 'Pclass'
plt.figure(figsize=(10, 7))
sns.boxplot(x='Pclass', y='Age', data=titanic_data)
plt.title('Age Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Age')

age_stats = titanic_data.groupby('Pclass')['Age'].describe()

# This shows that passengers in Pclass 1 tend to be older than in ther other classes
# This could indicate that the older you were, you were likely to have accumulated wealth,
# So they could afford the more expensive Fare.


# CLEANING THE DATA:

# Fill AGE column using method 'Imputation' -
# Using the mean of the AGE by PASSAGER Class:

plt.figure(figsize=(10, 7))
sns.boxplot(x='Pclass', y='Age', data=titanic_data)

# Creating a Function that will fill in the missing data from our Age column
# Based on the passanger class to the average age of each class.


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


titanic_data['Age'] = titanic_data[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(titanic_data.isnull(), yticklabels=False,
            cbar=False, cmap='coolwarm')

# Since there are alot of missing information for the Cabin Column
# Its best to drop it

titanic_data.drop('Cabin', axis=1, inplace=True)
titanic_data.dropna(inplace=True)

# Now data has no missing values 889 non-null

# DATA CLEANING:
# DEALING WITH CATEGORICAL DATA:
# Create new column to encode the SEX Column M-0, F-1

titanic_data['Embarked'].unique()
# And Embarked Column S, C, Q

sex = pd.get_dummies(titanic_data['Sex'], drop_first=True).astype(int)
embark = pd.get_dummies(titanic_data['Embarked'], drop_first=True).astype(int)
titanic_data = pd.concat([titanic_data, sex, embark], axis=1)

# DROPPING COLUMNS THAT WONT BE NEEDED FOR MACHINE LEARNING
titanic_data.drop(['PassengerId', 'Sex', "Embarked",
                  'Name', 'Ticket'], axis=1, inplace=True)
titanic_data.head(5)


# LOGISTICS REGRESSION:
# TRAIN AND USE MODEL TO PREDICT IF PASSENGERS SURVIVED

X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=45)

logmodel = LogisticRegression(max_iter=200)
logmodel.fit(X_train, y_train)

# Make predictions
predictions = logmodel.predict(X_test)
classification_report(y_test, predictions)
confusion_matrix(y_test, predictions)
