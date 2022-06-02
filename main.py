import dash
from dash import dcc
from dash import html
import plotly.express as px
import uvicorn


# from dash.dependencies import Input, Output, State

import pandas as pd

# from mlflow.sklearn import load_model


###############################################################################
# APP INITIALIZATION
###############################################################################
app = dash.Dash(__name__)

# this is needed by gunicorn command in procfile
server = app.server


###############################################################################
# LOAD MODEL AND DATA
###############################################################################
# def __get_model():
#     """Load the saved model for prediction.

#     Returns:
#         _type_: Model
#     """
#     model_path = "models/deployment_ann"
#     model = load_model(model_path)
#     return model


# def __get_data():
#     """Load and preprocess sample test dataset.

#     Returns:
#         _type_: X_test
#     """
#     path = "data/processed/"
#     X_test = pd.read_csv(path + "X_test.csv")
#     return X_test


# def __get_predictions(model, X_test):
#     """Query predictions

#     Args:
#         model (_type_): _description_
#         X_test (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     y_proba = model.predict_proba(X_test)
#     return y_proba


# model = __get_model()
# X_test = __get_data()
# y_proba = __get_predictions(model, X_test)

path = "data/"
X_test = pd.read_csv(path + "X_test.csv")
y_test = pd.read_csv(path + "y_test.csv")
y_pred = pd.read_csv(path + "y_pred.csv")
y_pred = 1 - y_pred

###############################################################################
# PLOTS
###############################################################################

fig = px.histogram(x=X_test.smart_999, color=y_pred["0"])


###############################################################################
# LAYOUT
###############################################################################
app.layout = html.Div(
    [
        html.H2(
            id="title",
            children="Predictive maintenance of HDDs in data centers",
        ),
        html.Div(id="textarea-state-example-output", style={"whiteSpace": "pre-line"}),
        dcc.Graph(id="bar-chart", figure=fig),
    ]
)


###############################################################################
# INTERACTION CALLBACKS
###############################################################################
# https://dash.plotly.com/basic-callbacks
# @app.callback(
#     [
#         Output("textarea-state-example-output", "children"),
#         Output("bar-chart", "figure"),
#     ],
#     Input("textarea-state-example-button", "n_clicks"),
#     State("textarea-state-example", "value"),
# )
# def update_output(n_clicks, value):
#     fig = get_figure(X_test.smart_999, y_proba)
#     if n_clicks > 0:
#         if 0 < len(value) < 10:
#             text = "you said: " + value
#             fig = get_figure(X_test.smart_999, y_proba)
#             return text, fig
#         else:
#             return "Please add a text between 0 and 10 characters!", fig
#     else:
#         return "", fig


# Add the server clause:
if __name__ == "__main__":
    app.run_server()
