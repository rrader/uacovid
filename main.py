import csv
import pathlib
import re
from datetime import datetime

import mwclient
import numpy
import pandas
from mwtemplates import TemplateEditor

import matplotlib.pyplot as plt
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.trend import PolynomialTrendForecaster

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


def learn(series):
    model = AutoARIMA(sp=7)
    model.fit(series[:-30])
    return model


def build_training_data():
    return data['Total confirmed'].diff().asfreq('1D', method='ffill').dropna() + 1


def predict():
    fh = numpy.arange(30) + 1
    return model.predict(fh).rename('Predicted Total confirmed')


def build_plot(series, predicted):
    plt.figure()
    series.plot(color='g')
    predicted.plot(color='r')
    plt.legend()
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    plot_file = PLOTS_PATH.joinpath(datetime.today().strftime('%y_%m_%d.png'))
    plt.savefig(plot_file)
    plt.show()


if __name__ == '__main__':
    data = parse_wiki()
    series = build_training_data()
    model = learn(series)
    predicted = predict()
    build_plot(series, predicted)
