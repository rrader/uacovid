import csv
import pathlib
import re
from datetime import datetime

import mwclient
import numpy
import pandas
from mwtemplates import TemplateEditor

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import ReducedRegressionForecaster, TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformers.single_series.detrend import Detrender, Deseasonalizer

DATA_PATH = pathlib.Path(__file__).parent.joinpath('data')
PLOTS_PATH = pathlib.Path(__file__).parent.joinpath('plots')


def parse_wiki():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    csv_file = DATA_PATH.joinpath(datetime.today().strftime('%y_%m_%d.data'))
    if csv_file.exists():
        with csv_file.open() as f:
            return read_csv(f)

    site = mwclient.Site('en.wikipedia.org')
    page = site.pages[u'Template:COVID-19_pandemic_data/Ukraine_medical_cases_chart']
    assert page.exists
    template_data = page.text()
    te = TemplateEditor(template_data)
    template = te.templates['Medical cases chart'][0]
    csv_data = template.parameters['data'].value
    csv_data = re.sub("(<!--.*?-->)", "", csv_data, flags=re.DOTALL).strip()

    with csv_file.open('w') as f:
        f.write(csv_data)
        return read_csv(csv_data.split('\n'))


def read_row(row):
    row['Date'] = datetime.strptime(row['Date'], '%Y-%m-%d')
    return row


def read_csv(f):
    return pandas.DataFrame([
        read_row(row) for row in
        csv.DictReader(
            f,
            delimiter=';',
            fieldnames=['Date', 'Deaths', 'Recoveries', 'Total confirmed']
        )
        if row['Date']
    ]).set_index(
        'Date', drop=False
    ).astype({'Total confirmed': 'int32'})


def learn(series_data):
    model = TransformedTargetForecaster([
        ("deseasonalise", Deseasonalizer(model="multiplicative", sp=7)),
        ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=4))),
        ("forecast", PolynomialTrendForecaster(degree=4))
    ])
    model.fit(series_data[:-2])
    return model


def build_training_data():
    return data['Total confirmed'].diff().asfreq('1D', method='ffill').dropna() + 1


def predict(model):
    fh = numpy.arange(12) + 1
    return model.predict(fh).rename('Predicted Total confirmed')


def build_plot(series, series2, predicted):
    plt.figure()
    series.plot(color='g')
    series2.plot(color='y')
    predicted.plot(color='r')

    plt.legend()
    # PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    # plot_file = PLOTS_PATH.joinpath(datetime.today().strftime('%y_%m_%d.png'))
    # plt.savefig(plot_file)
    plt.grid()
    plt.show()


def week(lst):
    r = []
    date = None
    for i, val in lst.iteritems():
        r.append(val)
        if date is None:
            date = i
        if len(r) == 7:
            yield date, numpy.mean(r)
            r = []
            date = None
    if r:
        yield date, numpy.mean(r)


if __name__ == '__main__':
    data = parse_wiki()
    series = build_training_data()
    series2 = pandas.DataFrame(week(series)).set_index(0)[1].asfreq('W', method='ffill').rename('Average by week')
    print(pandas.DataFrame(week(series)).set_index(0)[1].asfreq('W', method='ffill').to_csv())
    model_v = learn(series2)
    predicted = predict(model_v)
    build_plot(series, series2, predicted)
