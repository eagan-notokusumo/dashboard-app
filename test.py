import base64
import datetime
import io
from operator import itemgetter
import itertools
from os import remove

import dash
from dash import html, Input, Output, State, dcc, dash_table
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import dash_cytoscape as cyto
from algorithms import Graph, dijkstra_algorithm, path
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import copy



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
    # df['OA/M3'] = df['OA'].div(df['Kapasitas'].values).astype(int)
    # df['Index'] = df['Source'].astype(str) + df['Tipe_Kendaraan'] + df['Tujuan']
    df['Index'] = df['Route'] + df['Vendor_ID']
    # groups = df.groupby(['Index','Tujuan','Product_Code']).apply(list)
    # df = groups.reset_index()
    print(df)
    
    if 'Product_Code' in df.columns():
        children = html.Div(
            [
                html.H5(filename),
                html.H6(datetime.datetime.fromtimestamp(date)),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id='data-format-choice',
                                    clearable=False,
                                    options=[
                                        'Minimum Cost',
                                        'Average Cost',
                                        'Maximum Cost',
                                    ],
                                    placeholder='Enter cost data format'
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
                                        for i in df['Destination'].unique()
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
                                    clearable=True,
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
                                dcc.Dropdown(
                                    id='num-paths',
                                    clearable=False,
                                    options=[
                                        i for i in range(1,100)
                                    ],
                                    placeholder='Enter number of paths'
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
                    id='source-data',
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
    else:
        children = html.Div(
            [
                html.H5(filename),
                html.H6(datetime.datetime.fromtimestamp(date)),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id='data-format-choice',
                                    clearable=False,
                                    options=[
                                        'Minimum Cost',
                                        'Average Cost',
                                        'Maximum Cost',
                                    ],
                                    placeholder='Enter cost data format'
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
                                        for i in df['Destination'].unique()
                                    ],
                                    placeholder='Enter destination'
                                )
                            ],
                            className='two columns'
                        ),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id='num-paths',
                                    clearable=False,
                                    options=[
                                        i for i in range(1,100)
                                    ],
                                    placeholder='Enter number of paths'
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
                    id='source-data',
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

def yen_algorithm(init_graph, nodes, start_node, end_node, max_k):
    graph = Graph(nodes, init_graph)
    previous_nodes, shortest_path = dijkstra_algorithm(graph, start_node)

    A = [{
        'cost': shortest_path[end_node],
        'path_': path(previous_nodes, start_node, end_node)
    }]

    B = []

    if not A[0]['path_']:
        return A

    for k in range(1, max_k):
        for i in range(0,len(A[k-1]['path_']) - 1, 2):
            spur_node = A[k-1]['path_'][i]
            root_path = A[k-1]['path_'][:i]

            removed_edges = []
            pp = sorted([A[j]['path_'][:i] for j in range(len(A)-1)])
            pp_unique = list(pp for pp,_ in itertools.groupby(pp))

            for path_k in A:
                current_path = path_k['path_']
                # print(current_path[i+1])
                if i < len(current_path) - 1:
                    if len(current_path[i+1]) > 7 and current_path[i+1] in nodes: # removing transient nodes
                        # print(current_path[i+1])
                        nodes.remove(current_path[i+1])
                        # print(current_path[:i])
                        if len(current_path) - 1 > i and root_path == current_path[:i]:
                        # print(len(nodes))
                            cost = init_graph[current_path[i]].pop(current_path[i+1])
                            init_graph[current_path[i+1]].pop(current_path[i])
                            if cost == -1:
                                continue
                            removed_edges.append([current_path[i], current_path[i+1], cost])
                        
                    elif any(pp_unique):
                        for previous_path in pp_unique:
                            p0 = len(previous_path[:i-2])
                            p1 = len(previous_path[:i-1])
                            if current_path[:i] not in pp_unique:
                                if previous_path[p1] not in init_graph[previous_path[p0]].keys():
                                    continue
                                else:
                                    cost2 = init_graph[previous_path[p0]].pop(previous_path[p1])
                                    init_graph[previous_path[p1]].pop(previous_path[p0])
                                    removed_edges.append([previous_path[p0], previous_path[p1], cost2])
                            elif current_path[:i] == root_path and previous_path != current_path[:i]:
                                if previous_path[p1] not in init_graph[previous_path[p0]].keys():
                                    continue
                                else:
                                    cost3 = init_graph[previous_path[p0]].pop(previous_path[p1])
                                    init_graph[previous_path[p1]].pop(previous_path[p0])
                                    removed_edges.append([previous_path[p0], previous_path[p1], cost2])

                            

                    # print(removed_edges)
                    # print(init_graph)

                    # TODO: issue that nodes disappear

            graph_spur = Graph(nodes, init_graph)
            pn_init, sp_init = dijkstra_algorithm(graph_spur, start_node)
            pn_spur, sp_spur = dijkstra_algorithm(graph_spur, spur_node)
            spur_path = {
                'cost': sp_spur[end_node],
                'path_': path(pn_spur, spur_node, end_node)
            }
            
            if spur_path['path_']:
                total_path = root_path + spur_path['path_']
                total_cost = sp_init[spur_node] + spur_path['cost'] #TODO: the issue is shortest path is the oldest version
                potential_k = {'cost': total_cost, 'path_': total_path}

                if not (potential_k in B):
                    B.append(potential_k)

            for edge in removed_edges:
                init_graph[edge[0]][edge[1]] = edge[2]
                init_graph[edge[1]][edge[0]] = edge[2]
                if edge[1] not in nodes:
                    nodes.append(edge[1])

        if len(B):
            B = sorted(B, key=itemgetter('cost'))
            A.append(B[0])
            B.pop(0)
        else:
            break
    return A

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
    Output('source-data', 'data'),
    Output('node-data', 'data'),
    Output('init-graph', 'data'),
    Input('data-button', 'n_clicks'),
    State('stored-data', 'data'),
    # State('product-choice', 'value'),
    State('data-format-choice', 'value')
)
def update_data(n, data, format_data):
    # print(start)
    # print(format_data)
    if n is None or data is None or format_data is None:
        raise PreventUpdate
    
    df = pd.DataFrame.from_records(data)
    # df = df[df['Product_Code'] == prod]

    if format_data == 'Minimum Cost':
        groups = df.groupby(['Index', 'Destination']).min().apply(list)

    elif format_data == 'Average Cost':
        groups = df.groupby(['Shipping_Type', 'Destination']).mean().astype(int).apply(list)

    elif format_data == 'Maximum Cost':
        groups = df.groupby(['Index','Destination']).max().apply(list)

    df = groups.reset_index()
    df['OA/M3'] = df['OA'].div(df['Kapasitas'].values).astype(int)
    print(df)
    # TODO: reconsider how data is inputted i.e. source -> routes -> dest -> routes -> dest
    # TODO: skip STO destinations, use directly in routes, and only have source, main dest.

    src_init = ['Source']
    src =  [node for node in df['Source'].unique()]
    vehicles = [node for node in df['Index']]
    dstns = [node for node in df['Destination'].unique()]
    nodes = src_init + src + vehicles + dstns

    init_graph = {}
    for node in nodes:
        init_graph[node] = {}

    for node in tppt:
        filtered = df[df['Tppt'] == node]
        init_graph[nodes[0]][node] = 1
        query_vehicle = [x for x in filtered['Index']]

        for vehicle in query_vehicle:
            init_graph[node][vehicle] = 1

    for dstn in dstns:
        filtered = df[df['Tujuan'] == dstn]
        query_vehicle = [x for x in filtered['Index']]

        for vehicle in query_vehicle:
            index = query_vehicle.index(vehicle)
            init_graph[vehicle][dstn] = filtered.iloc[index, -1]
                    
    # print(init_graph)
    nodes_dict = {i: nodes[i] for i in range(0,len(nodes))}
    sources_dict = {i: tppt[i] for i in range(0,len(tppt))}

    #TODO: change output-table

    # print(df)
    return sources_dict, nodes_dict, init_graph

# TODO: algorithm callback
@app.callback(
    Output('output-algorithm', 'children'),
    Input('algorithm-button', 'n_clicks'),
    State('node-data', 'data'),
    State('init-graph', 'data'),
    State('num-paths', 'value'),
    State('destination-choice', 'value'),
    State('source-data', 'data'),
)
def algorithm(n, node_data, graph_data, num_paths, dest_node, source_data): # Yen's algorithm
    # print(data)
    if n is None or node_data is None or graph_data is None:
        raise PreventUpdate

    srcs = list(source_data.values())
    costs = {}
    paths = {}
    for src in srcs:
        local_graph = copy.deepcopy(graph_data)
        nodes = list(node_data.values())
        costs[src] = []
        paths[src] = []

        local_graph['Source'].clear()

        local_graph['Source'][src] = 1
        local_graph[src]['Source'] = 1

        data_node = [x for x in nodes if x.endswith(src) and len(x)  > 6]
        for node in data_node:
            # local_graph[src].pop(node)
            # nodes.remove(node)
            local_graph[node].clear()

        A = yen_algorithm(local_graph, nodes, start_node=src, end_node=dest_node, max_k=num_paths)

        for route in A:
            costs[src].append(route['cost'])
            paths[src].append(route['path_'])
        
        # for node in data_node:
        #     nodes.append(node)

    costs_df = pd.DataFrame.from_dict(costs).reset_index()
    # costs2 = costs_df.transpose().reset_index()
    paths_df = pd.DataFrame.from_dict(paths).reset_index()

    # print(costs_df)
    # print(len(costs_df))
    # print(paths_df)

    colors = ['rgb(26,118,255)', 'rgb(46,80,255)', 'rgb(55,83,109)', 'rgb(70,255,255)']
    line_colors = ['lightsalmon', 'indianred', 'orange']
    
    cost_total = [i for i in costs_df['3082']] + [i for i in costs_df['3083']]
    paths_total = [i for i in paths_df['3082']] + [i for i in paths_df['3083']]

    dict_data = {'Path': paths_total, 'Costs': cost_total}
    
    df = pd.DataFrame(dict_data)
    sorted_df = df.sort_values('Costs')
    print(sorted_df)

    fig = make_subplots(
        rows=1,
        cols=2,
    )
    
    for value in costs_df:
        if value == 'index':
            continue

        fig.add_trace(
            go.Bar(
                x=[i for i in costs_df['index']],
                y=[i for i in costs_df[value]],
                name=value,
                marker_color=colors[costs_df.columns.get_loc(value)],
                hovertext=[i for i in paths_df[value]],
            ),
            row=1,
            col=1,
        ),
        fig.add_trace(
            go.Scatter(
                x=[i for i in costs_df['index']],
                y=[i for i in costs_df[value]],
                name=value,
                marker_color=colors[costs_df.columns.get_loc(value)],
                hovertext=[i for i in paths_df[value]],
                fill='tozeroy',
            ),
            row=1,
            col=2,
        )
    
    fig.update_layout(
        barmode='group',
        xaxis=dict(
            title='Top Optimal Routes',
            tickfont_size=14,
        ),
        yaxis=dict(
            title='OA/M3 (Thousands of Rupiah)',
            titlefont_size=16,
            tickfont_size=14,
        ),
        bargap=0.15,
        bargroupgap=0.1,
    )


    
    # print(costs)
    # print(paths)
    return html.Div(
        [
            html.Div(
                [
                    dcc.Graph(figure=fig)
                ]
            )
        ]
    )

# @app.callback(
#     Output('output-algorithm', 'children'),
#     Input('destination-choice', 'value'),
#     State('shortest-path', 'data'),
#     State('previous-nodes', 'data')
# )
# def output_algorithm(dest_choice, shortest_path, previous_nodes):
#     print(dest_choice)
#     if shortest_path is None or previous_nodes is None:
#         raise PreventUpdate

#     path = []
#     node = dest_choice

#     while node != 'Source':
#         path.append(node)
#         node = previous_nodes[node]

#     path.append('Source')

#     return html.Div(
#         [
#             html.Div(
#                 html.P('Best path value: {}'.format(shortest_path[dest_choice]))
#             ),
#             html.Div(
#                 html.P(" -> ".join(reversed(path)))
#             )
#         ]
#     )
if __name__ == '__main__':
    app.run_server(debug=True)


#TODO: fix layouts
#TODO: remove individual node option (even to an entire BIG node), figure out how to affect init_graph
#TODO: add comparisons for multiple routes
#TODO: add bar graphs to show price comparisons x axis: sources, y axis: prices