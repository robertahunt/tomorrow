import os
import holidays

import pandas as pd
from models import models

config = dict(data_path = 'DK-DK2.csv')

def getStartEnd(date):
    if date == 'next24':
        start = pd.Timestamp.now(tz='UTC').round('1h') + pd.DateOffset(hours=1)
        
    else:
        try: 
            start = pd.Timestamp(date, tz='Europe/Copenhagen')
        except:
            raise 'Could not parse date input: "%s"'%date
    end = start + pd.DateOffset(days=1)
    return start.tz_convert('UTC'), end.tz_convert('UTC')

class carbonIntensityForecaster():
    def __init__(self, modelName='production',config=config):
        self.modelName = modelName
        self.config = config
        self.data = self.load_data()
        self.model = models[modelName]()
        
    def load_data(self):
        fp = self.config['data_path']
        assert os.path.exists(fp), 'Data File not found at %s'%fp
        try:
            df = pd.read_csv(fp)
        except:
            raise 'Error reading data file in %s'%fp
            
        df['yesterday_carbon_intensity_avg'] = df['carbon_intensity_avg'].shift(24)
        df['2dayold_carbon_intensity_avg'] = df['carbon_intensity_avg'].shift(24*2)
        df['weekold_carbon_intensity_avg'] = df['carbon_intensity_avg'].shift(24*7)
        
        #parse timestamp into index
        df.index = pd.to_datetime(df['datetime'])
        df.index.name = 'UTC'

        #get holidays, day of week, local hour including dst, 
        df['local_time'] = df.tz_convert('Europe/Copenhagen').index
        dk_holidays = holidays.DK()
        df['holiday?'] = df.local_time.map(lambda x: x in dk_holidays)

        df['weekday'] = df.local_time.map(lambda x: x.weekday)
        df['weekend?'] = df.weekday.map(lambda x: x < 5)
        return df
    
    def forecast(self, date='next24'):
        start, end = getStartEnd(date)
        datetimes = pd.date_range(start,end,freq='H',closed='right')
        
        model_df = self.data[self.model.xcols + self.model.ycol]
        
        #drop nans
        model_df = model_df.dropna(subset=self.model.xcols)
        
        #split into x, y
        x_df = model_df[self.model.xcols]
        y_df = model_df[self.model.ycol]
        
        #check that we have features for all datetimes.
        for dt in datetimes:
            assert dt in x_df.index, 'Error, some data missing from datafile for %s, forecast could not be produced'%dt
        
        #split into train, test
        train_x = x_df[:start].values
        train_y = y_df[:start].values
        test_x = x_df[start:end].values
        test_y = y_df[start:end].values
        
        self.model.fit(train_x,train_y)
        y_pred = self.model.predict(test_x)
        y_pred = pd.Series(y_pred,index=y_df[start:end].index)
        return y_pred