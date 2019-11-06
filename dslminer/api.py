from flask import Flask
from flask import request
from . import timeseries
app = Flask(__name__)

# forecast indicator data
@app.route('/forecast/<indicatorid>/')
def show_user_profile(indicatorid):
    periodspan = request.args.get('periodspan') # the number of dependent values to be generated based on periodtype
    periodtype = request.args.get('periodtype') # can yealy 'Y' or montly 'M'
    startyearmonth = request.args.get('startyearmonth') # start scope date for our dataset
    endyearmonth = request.args.get('endyearmonth') # end scope date for our dataset
    p=timeseries.predictor()
    p.predict()
    # show the user profile for that user
    return "{data}"