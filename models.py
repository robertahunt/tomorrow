from sklearn.neural_network import MLPRegressor

class productionModel():
    def __init__(self):
        self.xcols = ['latest_forecasted_dewpoint_avg', 'latest_forecasted_precipitation_avg',
               'latest_forecasted_solar_avg', 'latest_forecasted_temperature_avg',
               'latest_forecasted_wind_x_avg', 'latest_forecasted_wind_y_avg',
               'latest_forecasted_price_avg', 'latest_forecasted_production_avg',
               'latest_forecasted_consumption_avg',
               'latest_forecasted_power_net_import_DE_avg',
               'latest_forecasted_power_net_import_DK-DK1_avg',
               'latest_forecasted_power_net_import_SE-SE4_avg',
               'latest_forecasted_production_solar_avg',
               'latest_forecasted_production_wind_avg', '2dayold_carbon_intensity_avg',
               'weekold_carbon_intensity_avg', 'holiday?', 
               'weekend?']
        self.ycol = ['carbon_intensity_avg']
    
    def fit(self, x, y):
        self.model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=42)
        self.model.fit(x, y)
        
    def predict(self, x):
        return self.model.predict(x)
    
models = dict(production = productionModel)