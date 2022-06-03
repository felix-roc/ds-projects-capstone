import pandas as pd
import plotly.express as px
import dash
from dash import html, dcc, callback, Output, Input

# APP initialization
app = dash.Dash(__name__)
server = app.server  # this is needed by gunicorn command in procfile

# Load data
path = "data/"
X_test = pd.read_csv(path + "test_data_final.csv")
y_pred = pd.read_csv(path + "pred_test_data.csv")
y_pred = y_pred.prediction.astype(bool)
# Store unique serial numbers and smart numbers that contain values
serials = X_test.serial_number.unique()
X_test = X_test.dropna(axis=1)
smarts = X_test.columns[X_test.columns.str.contains("raw")].to_numpy()

# Initialize figure
fig = px.scatter(y=X_test.smart_197_raw, color=y_pred)

# Layout
app.layout = html.Div(
    [
        html.H2(
            id="title",
            children="Predictive maintenance of HDDs in data centers",
            style={"textAlign": "center"},
        ),
        html.H4(id="select_serial", children="Please select one HDD:"),
        dcc.Dropdown(serials, serials[2], multi=False, id="serial-dropdown"),
        html.H4(
            id="select_smart", children="Please select one S.M.A.R.T. feature to plot:"
        ),
        dcc.Dropdown(smarts, smarts[18], multi=False, id="smart-dropdown"),
        html.H4(id="whitespace", children=" "),
        dcc.Graph(id="chart", figure=fig),
    ]
)


@callback(
    Output("chart", "figure"),
    [Input("serial-dropdown", "value"), Input("smart-dropdown", "value")],
)
def update_graph(serial_selected, smart_selected):
    data_selected = X_test[X_test.serial_number == serial_selected]
    pred_selected = y_pred[data_selected.index]
    fig = px.scatter(
        y=data_selected[smart_selected],
        color=pred_selected,
        labels={"x": "Days", "y": str(smart_selected), "color": "Predicted Failure"},
    )
    return fig


if __name__ == "__main__":
    app.run_server()
