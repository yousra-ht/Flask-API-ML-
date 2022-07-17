from array import array
from asyncio.windows_events import NULL
from flask import Flask , request , jsonify
from datetime import datetime , date 
import numpy as np
import pickle
from dateutil.relativedelta import relativedelta 
app = Flask(__name__)


today = date.today()
todayfake =  datetime.strptime("2022-02-01", "%Y-%m-%d")
currentYear = datetime.now().year
currentMonth = datetime.now().month
delta = relativedelta(months=5)
modele5mois = pickle.load(open('../ML/model5mois.pkl', 'rb'))
modele1mois = pickle.load(open('../ML/model1mois.pkl', 'rb'))
modele3mois = pickle.load(open('../ML/model3mois.pkl', 'rb'))
@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/Test',  methods = ['POST'])
def Post():
    data = request.json
    return jsonify(data['test'])

@app.route('/predict', methods = ['POST'])
def churn():
    data = request.json
    Problem = data['Problem']
    Gender = data['Gender']
    Payment = data['Payment']
    city = data['RestaurantInfo']['city']
    creationYear = data['RestaurantInfo']['creationYear']
    type = data['RestaurantInfo']['type']
    Amount =  data['Amount'] 
    AmountListe = [0.0,0.0,0.0,0.0,0.0]
    
    for i in range(len(Amount)):
        if ( Amount[i]['year'] == str ((todayfake - relativedelta(months=5)).year) and  Amount[i]['month'] == str( (todayfake- relativedelta(months=5)).month ) ): 
                AmountListe[0]= float( (Amount[i]['amount']))
        if ( Amount[i]['year'] == str ((todayfake - relativedelta(months=4)).year) and  Amount[i]['month'] == str( (todayfake- relativedelta(months=4)).month ) ): 
                AmountListe[1]= float( (Amount[i]['amount']))
        if ( Amount[i]['year'] == str ((todayfake - relativedelta(months=3)).year) and  Amount[i]['month'] == str( (todayfake- relativedelta(months=3)).month ) ): 
                AmountListe[2]= float( (Amount[i]['amount']))
        if ( Amount[i]['year'] == str ((todayfake - relativedelta(months=2)).year) and  Amount[i]['month'] == str( (todayfake- relativedelta(months=2)).month ) ): 
                AmountListe[3]= float( (Amount[i]['amount']))
        if ( Amount[i]['year'] == str ((todayfake - relativedelta(months=1)).year) and  Amount[i]['month'] == str( (todayfake- relativedelta(months=1)).month ) ): 
                AmountListe[4]= float( (Amount[i]['amount']))
         
    if Gender == 'u' : 
        Gender = float(2) 
    elif Gender == 'f'  :
        Gender = float(0)
    else  :
        Gender = float(1)  

    if city.upper() ==   'PARIS' :
        city = float(2)
    elif city.upper() ==  'VITRY SUR SEINE' :
        city = float(4)
    elif city.upper() ==  'BOULOGNE-BILLANCOURT' :
        city = float(1)
    elif city.upper() ==  'AUBERVILLIERS' :
        city = float(0)
    else : 
        city = float(3)

    if  not Payment :
        Payment = float(0)
    else : 

       
        if Payment['paymentMethod'].upper() == 'A CREDIT' : 
                Payment = float(0) 
        elif Payment['paymentMethod'].upper() == 'CB' :
                Payment = float(1)
        elif Payment['paymentMethod'].upper() == 'CHEQUE' :
                Payment = float(1)
    

    if type is None :
        type = float(0)
    
    if not Problem : 
        Problem = float(0)
        print(Problem)
    else : 
        Problem = float(Problem['problem'])
        print(Problem)
       

    if AmountListe[0] == 0.0 and AmountListe[1] == 0.0 and AmountListe[2] == 0.0 and AmountListe[3] == 0.0 :
        feature = [city ,float(creationYear),Gender ,Problem, float(type) , Payment, AmountListe[4]] 
        featureArray  = [np.array(feature)]
        prediction = modele1mois.predict(featureArray)
    elif AmountListe[0] == 0.0 and AmountListe[1] == 0.0 and AmountListe[2] == 0.0 :
        feature = [city ,float(creationYear),Gender ,Problem, float(type) , Payment, AmountListe[3] ,AmountListe[4]]
        featureArray  = [np.array(feature)] 
        prediction = modele3mois.predict(featureArray)
    else :
        feature = [city ,float(creationYear),Gender ,Problem, float(type) , Payment, AmountListe[0] ,AmountListe[1],AmountListe[2],AmountListe[3], AmountListe[4]] 
        featureArray  = [np.array(feature)]
        prediction = modele5mois.predict(featureArray)




    
    print(feature)
   

    return jsonify( {'prediction':str (prediction[0]) }) 
   
