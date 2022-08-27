from dash import html
import dash_bootstrap_components as dbc

def NavBar():
    layout = html.Div(
        [
            dbc.NavbarSimple(
                [
                    dbc.NavItem(dbc.NavLink('How to', href='/how-to')),
                    dbc.NavItem(dbc.NavLink('Dashboard', href='/dashboard')),
                ],
                brand='Network Optimization App',
                brand_href='/how-to',
                color='dark',
                dark=True,
            ),
        ]
    )
    return layout