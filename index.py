import base64
import datetime
import io

import dash
from dash import html, Input, Output, State, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
import dash_cytoscape as cyto
from pyrsistent import l
from algorithms import yen_algorithm

import pandas as pd
import numpy as np
import pprint



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width"}
    ],
    suppress_callback_exceptions=True
)

def build_banner():
    return html.Div(
            id='banner',
            className='banner',
            children=[
                html.Div([
                    html.Div(
                        id='banner-text',
                        children=[
                            html.H5('Network Optimization Dashboard'),
                            html.H6('Input Control and Route Analysis'),
                        ],
                    ),
                    html.Div(
                        id='banner-logo',
                        children=[
                            html.Button(
                            id='learn-more-button',
                            children='learn-more',
                            n_clicks=0
                            ),
                        ]
                    )
                ],
                className='row')
            ],
    )

def build_tabs():
    return html.Div(
        id='tabs',
        className='tabs',
        children=[
            dcc.Tabs(
                id='app-tabs',
                value='tab1',
                className='custom-tabs',
                children=[
                    dcc.Tab(
                        id='input-tab',
                        label='Model settings',
                        value='tab1',
                        className='custom-tab',
                        selected_className='custom-tab--selected',
                    ),
                    dcc.Tab(
                        id='model-tab',
                        label='Model Output Dashboard',
                        value='tab2',
                        className='custom-tab',
                        selected_className='custom-tab--selected',
                    ),
                ],
            ),
        ],
    )

def build_tab1():
    return [
        html.Div(
            [
                html.Div(
                    id='set-input-intro-container',
                    children=html.P(
                        'Upload data, and enter inputs for model specifications, and choice of single source or multi-source evaluation.'
                    ),
                ),
                html.Div(
                    id='setting-menu',
                    children=[
                        html.Div(
                            id='menu-left',
                            children=[
                                html.Label(
                                    id='upload-data-title',
                                    children='Upload Data',
                                ),
                                html.Br(),
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div(
                                        [
                                            'Drag and drop files or',
                                            html.A('Select Files')
                                        ]
                                    ),
                                    style={
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'autosize': True
                                    },
                                    multiple=True,
                                ),
                            ]
                        ),
                    ],
                ),
                html.Div(
                    id='value-setter-menu',
                    children=[
                        html.Div('value-setter-panel'),
                        html.Br(),
                        html.Div(
                            id='button-div',
                            children=[
                                html.Button("Update Data", id='update-data-btn'),
                                html.Button(
                                    'Generate Model', 
                                    id='generate-model-btn',
                                    n_clicks=0,
                                ),
                            ],
                        ),
                        html.Div(
                            id='value-setter-output',
                            className='output-datatable-raw'
                        )
                    ],
                ),
            ],
        ),
    ]

def build_value_setter_line(line_num, label, value):
    return html.Div(
        id=line_num,
        children=[
            html.Label(label, className='six columns'),
            html.Div(value, className='six columns'),
        ],
        className='row'
    )

def quick_stats_panel():
    return html.Div(
        id='quick-stats',
        className='row',
        children=[]
    )

def generate_modal():
    return html.Div(
        id='markdown',
        className='modal',
        children=[
            html.Div(
                id='markdown-container',
                className='markdown-container',
                children=[
                    html.Div(
                        className='close-container',
                        children=html.Button(
                            "Close",
                            id='markdown-close',
                            n_clicks=0,
                            className='closeButton',
                        ),
                    ),
                    html.Div(
                        className='markdown-text',
                        children=dcc.Markdown(
                            children=(
                                """
                                ###### What does this dashboard do?

                                This dashboard runs network optimization algorithms to process the most optimal routes 
                                dynamically given inputs, and displays results via graphs and charts, along with
                                the filtered data.

                                ###### How does this dashboard work?

                                First, click on 'Upload Data' to upload an excel/csv file containing all routes possible,
                                along with vehicle types and corresponding prices.

                                Different inputs can be selected to filter the data and generate the model.
                                Click on 'Update Data' and 'Generate Model' to launch the algorithm and present the results.
                                Some fields must be filled (destination, source(s), product), while others are optional.

                                ###### Article Reference
                                [Wikipedia Article](https://en.wikipedia.org/wiki/Yen%27s_algorithm)
                                """
                            )
                        )
                    )
                ]
            )
        ]
    )


def generate_section_banner(title):
    return html.Div(className='section-banner', children=title)

def build_top_panel():
    return html.Div(
        id='top-section-container',
        className='row',
        children=[
            html.Div(
                id='route-summary-section',
                className='eight columns',
                children=[
                    generate_section_banner('Optimal Paths Summary'),
                    generate_histogram(),
                ],
            ),
            html.Div(
                id='source-optimal-path',
                className='four columns',
                children=[
                    generate_section_banner('Top Overall Optimal Paths'),
                    generate_lists(),
                ],
            )
        ]
    )

def generate_histogram():
    return dcc.Graph(
        id='histogram',
    )

def generate_lists():
    return html.Div(
        id='lists',
    )

def generate_details():
    return html.Div(
        id='details',
    )

# def get_selection(selection):
#     xVal = []
#     yVal = []
#     xSelected = []

def build_bottom_panel():
    return html.Div(
        id='route-details-container',
        className='twelve columns',
        children=[
            generate_section_banner('route-details'),
            generate_details(),
        ]
    )

app.layout = html.Div(
    id='big-app-container',
    children=[
        build_banner(),
        html.Div(
            id='app-container',
            children=[
                build_tabs(),
                html.Div(
                    id='app-content',
                ),
            ],
        ),
        generate_modal(),
    ],
)
'''app.layout = html.Div([ # this code section taken from Dash docs https://dash.plotly.com/dash-core-components/upload
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
    
])'''
                
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
        html.Div(
            [
                build_value_setter_line(
                    'value-setter-panel-header',
                    'Specs',
                    'Set value',
                ),
                build_value_setter_line(
                    'value-setter-panel-source',
                    'Source: Single/Multi',
                    html.Div(
                        dcc.Dropdown(
                            id='start-choice',
                            clearable=True,
                            options=[
                                {'label': i, 'value': i}
                                for i in df['Tppt'].unique()
                            ],
                            placeholder='Enter source(s)',
                        ),
                    )
                ),
                build_value_setter_line(
                    'value-setter-panel-destination',
                    'Destination',
                    html.Div(
                        dcc.Dropdown(
                            id='dest-choice',
                            clearable=False,
                            options=[
                                {'label': i, 'value': i}
                                for i in df['Tujuan'].unique()
                            ],
                            placeholder='Enter destination'
                        ),
                    )
                ),
                build_value_setter_line(
                    'value-setter-panel-product',
                    'Product Code',
                    html.Div(
                        dcc.Dropdown(
                            id='product-choice',
                            clearable=False,
                            options=[
                                {'label': i, 'value': i}
                                for i in df['Product_Code'].unique()
                            ],
                            placeholder='Enter product code',
                        )
                    )
                ),
                build_value_setter_line(
                    'value-setter-panel-path',
                    'Number of paths',
                    html.Div(
                        id='num-paths',
                        clearable=False,
                        options=[i for i in range(1,10)],
                        placeholder='Enter number of paths',
                    )
                ),
            ],
        ),
        dcc.Store(
            id='stored-data',
            data=df.to_dict('records')
        ),
        dcc.Store(
            id='source-data',
        ),
        dcc.Store(
            id='node-data',
        ),
        dcc.Store(
            id='init-graph',
        ),
        dcc.Store(
            id='costs-data'
        ),
        dcc.Store(
            id='paths-data'
        ),
    )
    
    return children


@app.callback(
    Output('app-content', 'children'),
    Input('app-tabs', 'value')
)
def render_tab_content(tab_switch):
    if tab_switch == 'tab1':
        return build_tab1()
    
    return html.Div(
        id='status-container',
        children=[
            quick_stats_panel(),
            html.Div(
                id='graphs-container',
                children=[
                    build_top_panel(),
                    build_bottom_panel(),
                ]
            )
        ]
    )
@app.callback(
    Output('markdown', 'style'),
    [
        Input('learn-more-button', 'n_clicks'),
        Input('markdown-close', 'n_clicks')
    ],
)
def update_click_output(button_click, close_click):
    if button_click is None or close_click is None:
        raise PreventUpdate

    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if prop_id == 'learn-more-button':
            return {'display': 'block'}
    
    return {'display': 'none'}


@app.callback(
    Output('value-setter-panel', 'children'),
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

@app.callback(
    Output('node-data', 'data'),
    Output('init-graph', 'data'),
    Output('source-data', 'data'),
    Input('update-data-btn', 'n_clicks'),
    # State('start-choice', 'value'),
    State('stored-data', 'data'),
    State('product-choice', 'value'),
)
def update_data(n, data, prod):
    # print(start)
    if n is None or data is None:
        raise PreventUpdate
    
    df = pd.DataFrame.from_records(data)
    # df = df[df['Tujuan'] == dest]
    df = df[df['Product_Code'] == prod]
    
    src = ['Source'] + [node for node in df['Tppt'].unique()]
    vehicles = [node for node in df['Index']]
    dstns = [node for node in df['Tujuan'].unique()]
    nodes = src + vehicles + dstns

    init_graph = {}
    for node in nodes:
        init_graph[node] = {}

    for vehicle in vehicles:
        init_graph[nodes[0]][vehicle] = 1

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
        
    # print(init_graph)
    nodes_dict = {i: nodes[i] for i in range(0,len(nodes))}
    sources_dict = {i: src[i] for i in range(0,len(src))}

    return nodes_dict, init_graph, sources_dict

# TODO: algorithm callback
@app.callback(
    Output('costs-data', 'data'),
    Output('paths-data', 'data'),
    Input('generate-model-btn', 'n_clicks'),
    State('node-data', 'data'),
    State('init-graph', 'data'),
    State('num-paths', 'value'),
    State('destination-choice', 'value'),
    State('source-data', 'data'),
)
def algorithm(n, node_data, graph_data, num_paths, dest_node, source_data): # Yen's algorithm
    # print(data)
    if n is None or node_data is None or graph_data is None or source_data is None:
        raise PreventUpdate
    
    nodes = list(node_data.values())
    srcs = list(source_data.values())
    costs = {}
    paths = {}
    for src in srcs:
        costs[src] = []
        paths[src] = []
        removed_edges = []
        if src != 'Source':
            keys = list(graph_data[nodes[0]].keys())
            for key in keys:
                if src not in key:
                    val = graph_data[nodes[0]].pop(key)
                    graph_data[key].pop(nodes[0])
                    removed_edges.append([nodes[0], key, val])

        A = yen_algorithm(graph_data, nodes, start_node='Source', end_node=dest_node, max_k=num_paths)

        for route in A:
            costs[src].append(route['cost'])
            paths[src].append(route['path_'])

        for edge in removed_edges:
            graph_data[edge[0]][edge[1]] = edge[2]
            graph_data[edge[1]][edge[0]] = edge[2]


    return costs, paths

@app.callback(
    Output('histogram', 'figure'),
    Output('lists', 'children'),
    Input('app-tabs', 'value'),
    State('costs-data','data'),
    State('paths-data', 'data'),
)
def render_graphs(tab_switch, costs, paths):
    if tab_switch == 'tab1':
        return build_tab1()
    
    costs_df = pd.DataFrame.from_dict(costs).transpose().reset_index().drop(0)
    paths_df = pd.DataFrame.from_dict(paths).transpose().reset_index().drop(0)


    fig = px.histogram(
        costs_df, 
        x='index', 
        y=['0','1','2','3','4'], 
        barmode='group',
        text_auto='.5i',
        labels={'x': 'Sources', 'y': 'OA/m3'},
    )
    vals = []
    for i in range(len(costs_df.index)):
        row_val = costs_df.iloc[i, 1:3].to_list()
        vals.append(row_val)
    
    vals.sort()

    children = html.Div(
        [
            html.H6('Top 3 paths from each source ranked.'),
            html.Div(
                [html.Ol([html.Li(x) for x in vals])]
            ),
        ],
    )

    return fig, children

# @app.callback(
#     Output('details', 'children'),
#     Input('histogram', 'selectedData'),
#     Input('histogram', 'clickData'),
#     State('stored-data', 'data')
# )
# def generate_pages(value, clickData, data):




if __name__ == '__main__':
    app.run_server(debug=True)


#TODO: fix layouts
#TODO: remove individual node option (even to an entire BIG node), figure out how to affect init_graph
#TODO: add comparisons for multiple routes
#TODO: add bar graphs to show price comparisons x axis: sources, y axis: prices
