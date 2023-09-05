import pandas as pd
import sklearn
from flask import Flask, render_template, request
import pickle
import numpy as np

app=Flask(__name__) # home
model=pickle.load(open('completed_model.pkl','rb'))
testcsv = pd.read_csv("C:\\Users\\Prasad\\Desktop\\Intelligent_Property_Analyser-main\\updated_train.csv")
#define the default route
@app.route('/')
def home():
    var1=sorted(testcsv['MSSubClass'].unique())
    var2=sorted(testcsv['MSZoning'].unique())
    var3=sorted(testcsv['ExterCond'].unique())
    var4=sorted(testcsv['BsmtCond'].unique())
    var5=sorted(testcsv['HeatingQC'].unique())
    var6=sorted(testcsv['KitchenQual'].unique())
    var7=sorted(testcsv['FireplaceQu'].unique())
    var8=sorted(testcsv['PoolQC'].unique())
    var9=sorted(testcsv['Driveway'].unique())
    var10=sorted(testcsv['Utilities Available'].unique())
    return render_template('PricePredictor.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    global result
    if request.method == 'POST':
        var1 = str(request.form['MSSubClass'])
        var2 = str(request.form['MSZoning'])
        var3 = str(request.form['ExterCond'])
        var4 = str(request.form['BsmtCond'])
        var5 = str(request.form['HeatingQC'])
        var6 = str(request.form['KitchenQual'])
        var7 = str(request.form['FireplaceQu'])
        var8 = str(request.form['PoolQC'])
        var9 = str(request.form['Driveway'])
        var10 = str(request.form['Utilities Available'])
        #preddata = pd.DataFrame()
        preddata = pd.DataFrame([{'MSSubClass':var1,'MSZoning':var2,'ExterCond':var3,'BsmtCond':var4,'HeatingQC':var5,'KitchenQual':var6,'FireplaceQu':var7,'PoolQC':var8,'Driveway':var9,'Utilities Available':var10}])
        preddata['MSSubClass']=var1
        preddata['MSZoning']=var2
        preddata['ExterCond']=var3
        preddata['BsmtCond']=var4
        preddata['HeatingQC']=var5
        preddata['KitchenQual']=var6
        preddata['FireplaceQu']=var7
        preddata['PoolQC']=var8
        preddata['Driveway']=var9
        preddata['Utilities Available']=var10
        prediction=model.predict(pd.DataFrame['MSSubClass'],data=np.array([var1]))
        return str(np.round(prediction[0],2))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
