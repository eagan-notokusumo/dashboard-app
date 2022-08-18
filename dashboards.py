import base64
import datetime
import io

import dash
from dash import html, Input, Output, State, dcc
from dash.exceptions import PreventUpdate
import dash_table
import plotly.express as px
import dash_cytoscape as cyto
from dijsktra import Graph, dijkstra_algorithm

import pandas as pd
import numpy as np
import pprint


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

app.layout = html.Div([ # this code section taken from Dash docs https://dash.plotly.com/dash-core-components/upload
    html.H1(
        'Network Optimization Dashboard',
        style={'textAlign': 'center'}
    ),
    html.Hr(),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(
        [
            html.Div(
                id='output-layout',
            ),
        ]
    ),
    html.Hr(),
    html.Div(
        id='output-algorithm',
    ),
    html.Div(
        id='output-data', 
    ),
    html.Hr(),
    #TODO: algorithm div
    
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    # print(df.dtypes)
    df['Tppt'] = df['Tppt'].astype(str)
    df['OA'] = df['OA'].replace('\.', '', regex=True).astype(int)
    df['OA/M3'] = df['OA'].div(df['Kapasitas'].values).astype(int)
    df['Index'] = df['Tppt'].astype(str) + df['Tipe_Kendaraan'] + df['Tujuan']
    groups = df.groupby(['Index','Tujuan','Product_Code']).min().apply(list)
    df = groups.reset_index()

    children = html.Div(
        [
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                id='start-choice',
                                clearable=True,
                                # multi=True,
                                options=[
                                    {'label': i, 'value': i}
                                    for i in df['Tppt'].unique()
                                ],
                                placeholder='Enter source(s)',
                            )
                        ],
                        className='two columns'
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id='destination-choice',
                                clearable=False,
                                # value='BDG1',
                                options=[
                                    {'label': i, 'value': i}
                                    for i in df['Tujuan'].unique()
                                ],
                                placeholder='Enter destination'
                            )
                        ],
                        className='two columns'
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id='product-choice',
                                clearable=False,
                                # value='POCO5',
                                # multi=True,
                                options=[
                                    {'label': i, 'value': i}
                                    for i in df['Product_Code'].unique()
                                ],
                                placeholder='Enter product code'
                            )
                        ],
                        className='two columns'
                    ),
                    html.Div(
                        [
                            html.Button(
                                id='data-button',
                                children='Generate data'
                            )
                        ],
                        className='two columns'
                    ),
                    html.Div(
                        [
                            html.Button(
                                id='algorithm-button', 
                                children='Generate model'
                            )
                        ],
                        className='two columns'
                    )
                ],
                className='1 row'
            ),
            dcc.Store(
                id='stored-data',
                data=df.to_dict('records')
            ),
            dcc.Store(
                id='node-data',
            ),
            dcc.Store(
                id='init-graph',
            ),
            dcc.Store(
                id='shortest-path'
            ),
            dcc.Store(
                id='previous-nodes'
            )
            # html.Div('Raw Content'),
            # html.Pre(
            #     contents[0:200] + '...', 
            #     style={
            #         'whiteSpace': 'pre-wrap',
            #         'wordBreak': 'break-all'
            #     }
            # ),
        ]
    )
    return children

@app.callback(
    Output('output-layout', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_layout(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

# TODO: consider inputs of destination choice and product choice instead of generate data
@app.callback(
    Output('output-data', 'children'),
    Output('node-data', 'data'),
    Output('init-graph', 'data'),
    Input('data-button', 'n_clicks'),
    State('start-choice', 'value'),
    State('stored-data', 'data'),
    State('product-choice', 'value'),
)
def update_data(n, start, data, prod):
    print(start)
    if n is None or data is None:
        raise PreventUpdate
    
    df = pd.DataFrame.from_records(data)
    # df = df[df['Tujuan'] == dest]
    df = df[df['Product_Code'] == prod]
    
    # TODO: reconsider how data is inputted i.e. source -> routes -> dest -> routes -> dest
    # TODO: skip STO destinations, use directly in routes, and only have source, main dest.

    src = ['Source']
    vehicles = [node for node in df['Index']]
    dstns = [node for node in df['Tujuan'].unique()]
    nodes = src + vehicles + dstns

    init_graph = {}
    for node in nodes:
        init_graph[node] = {}

    for vehicle in vehicles:
        if start is None or start in vehicle:
            init_graph[nodes[0]][vehicle] = 1
        elif start is not None and start not in vehicle:
            continue

# fill out non source
# print(init_graph)
    for dstn in dstns:
        filtered = df[df['Tujuan'] == dstn]
        query_vehicle = [node for node in filtered['Index']]

        for vehicle in query_vehicle:
            index = query_vehicle.index(vehicle)
            init_graph[vehicle][dstn] = filtered.iloc[index, -1]

        for vehicle in vehicles:
            if vehicle.startswith(dstn):
                init_graph[dstn][vehicle] = 1
            else:
                continue
        
    print(init_graph)
    nodes_dict = {i: nodes[i] for i in range(0,len(nodes))}
    records = df.to_dict('records')

    children = html.Div(
        [
            dash_table.DataTable(
                id='output-table',
                data=records,
                columns=[
                    {'name': i, 'id': i}
                    for i in df.columns
                ]
            ),
        ]
    )
    # print(df)
    return children, nodes_dict, init_graph

# TODO: algorithm callback
@app.callback(
    Output('shortest-path', 'data'),
    Output('previous-nodes', 'data'),
    Input('algorithm-button', 'n_clicks'),
    State('node-data', 'data'),
    State('init-graph', 'data')
)
def algorithm(n, node_data, graph_data):
    # print(data)
    if n is None or node_data is None or graph_data is None:
        raise PreventUpdate
    
    nodes = list(node_data.values())
    # print(nodes)
    # print(type(nodes))
    graph = Graph(nodes, graph_data)
    previous_nodes, shortest_path = dijkstra_algorithm(graph, start_node='Source')
    # print(graph)
    print(previous_nodes)
    # print(type(previous_nodes))
    # print(shortest_path)
    # print(type(shortest_path))

    return shortest_path, previous_nodes

@app.callback(
    Output('output-algorithm', 'children'),
    Input('destination-choice', 'value'),
    State('shortest-path', 'data'),
    State('previous-nodes', 'data')
)
def output_algorithm(dest_choice, shortest_path, previous_nodes):
    print(dest_choice)
    if shortest_path is None or previous_nodes is None:
        raise PreventUpdate

    path = []
    node = dest_choice

    while node != 'Source':
        path.append(node)
        node = previous_nodes[node]

    path.append('Source')

    return html.Div(
        [
            html.Div(
                html.P('Best path value: {}'.format(shortest_path[dest_choice]))
            ),
            html.Div(
                html.P(" -> ".join(reversed(path)))
            )
        ]
    )
if __name__ == '__main__':
    app.run_server(debug=True)