import pandas
import kfold_template
from sklearn import tree


dataset = pandas.read_csv("temperature_data.csv")


dataset = pandas.get_dummies(dataset)


dataset = dataset.sample(frac=1).reset_index()
#randomly shuffle dataset- same dataset, different order- starts with not Jan 1


target = dataset["actual"].values
#this is the y variable we want in the "actual" column, values turns it into numbers, not code
data = dataset.drop("actual", axis =1)
#keeps only the explanatory variables and not the target, axis means columns
data = dataset.drop(["level_0", "actual"], axis =1)
#you cant drop both without a square bracket- drop takes an argument about which column to drop and a second about it being a column, so you can't separate drops with commas
#you can do data = dataset.drop("[level_0", "actual], axis =1)"
#wait to turn this into value because it drops names
feature_list= data.columns
data = data.values

print(feature_list)
print(target)
print(data)

machine = tree.DecisionTreeClassifier(criterion="gini",max_depth= 10)
#max_depth- how many times the tree can branch out or how many times you chop the data
#machine.fit(data, target)
#fit the machine to our x, y variables

return_values = kfold_template.run_kfold(machine, data, target, 4, True)
#splits into 4, True means continuous
print(return_values)

machine = tree.DecisionTreeClassifier(criterion="gini",max_depth= 10)
machine.fit(data, target)
feature_importances_row = machine.feature_importances_
print(feature_importances_row)
print(feature_list)

feature_zip = zip(feature_list, feature_importances_row)
print(feature_zip)
#you can't read this zip object
feature_importances = [ (feature, round(importance, 4))	for feature, importance in feature_zip]
#loop the feature zip, get the two columns feature and importance. Round importance to 4 decimals
feature_importances= sorted(feature_importances, key= lambda x: x[1] )
#to get the second item, which is importance- not the name, shows what is most important, which is the historical data- historical average is a good predictor
print(feature_importances) 
[ print('{:13}: {}'.format(*feature_importance)) for feature_importance in feature_importances]
#formats the findings nicely