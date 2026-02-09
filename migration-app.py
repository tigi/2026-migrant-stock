import pandas as pd
import dash
from dash import dcc, html
import dash_leaflet as dl
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import math
import numpy as np

# Stylesheets

vizro_bootstrap = "https://cdn.jsdelivr.net/gh/mckinsey/vizro@main/vizro-core/src/vizro/static/css/vizro-bootstrap.min.css?v=2"

# Constants
INCOME_COLORS = {
    "High income": "#2ca02c",
    "Upper middle income": "#1f77b4",
    "Lower middle income": "#ff7f0e",
    "Low income": "#d62728",
    "Unknown": "#7f7f7f"
}
INCOME_ORDER = ["High income", "Upper middle income", "Lower middle income", "Low income", "Unknown"]

SELECTED_YEAR = '2024'
#ALL YEARS WILL BE EQUIVALENT TO THE COLUMN NAMES WITH STOCK NUMBERS OF THE IMPORTED DATA
ALL_YEARS = [ '1990','1995','2000','2005','2010','2015','2020','2024']


# ============= DATA LOADING & PREPROCESSING =============

def load_and_preprocess_data():
    """Load and preprocess all data files with optimizations"""
    
    
    #read migration stock data, years 1990 -  2024
    df = pd.read_excel('undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx',
                            sheet_name='Table 1',
                            header=10,
                            usecols=[1,4,5,6,7,8,9,10,11,12,13,14])
    
    df.rename(columns={'Region, development group, country or area of destination': 'Destination',\
                        'Region, development group, country or area of origin': 'Origin'}, inplace=True)
    df.drop(columns=['Location code of destination','Location code of origin'], inplace=True)
    
    # convert all column names to strings
    df.columns = df.columns.astype(str)
    
    # Vectorized string cleaning (faster than apply)
    df['Origin'] = df['Origin'].str.strip().str.replace("*", "", regex=False)
    df['Destination'] = df['Destination'].str.strip().str.replace("*", "", regex=False)
    
    # Load country coordinates
    countries_df = pd.read_csv("countries.csv")
    known_countries = set(countries_df['name'].tolist())  # Use set for O(1) lookup
    
    # Load income classification
    
    income_df = pd.read_excel('OGHIST_2025_10_07.xlsx',
                             sheet_name='Country Analytical History',
                             header=10,
                             #isocode, countryname, 2024 classification (col 39)
                             usecols=[0,1,39])

    income_df.rename(columns={'Unnamed: 0': 'Iso', 'Unnamed: 1': 'Economy', 'Unnamed: 39':'Income group'}, inplace=True)

    # convert all column names to strings, just to be sure
    income_df.columns = income_df.columns.astype(str)

    #the imported income columns work with L,LM,UM,H
    income_groups = {'L': 'Low income','LM':'Lower middle income','UM':'Upper middle income','H':'High income'}

    income_df['Income group'] = income_df['Income group'].map(income_groups).fillna('Unknown')

    #if you know at forehand that countrynames are not equal and there's no ISO value

    country_names = {"Bolivia": "Bolivia (Plurinational State of)", "Venezuela": "Venezuela (Bolivarian Republic of)", "United States": "United States of America"}

    income_df['Economy'] = income_df['Economy'].replace(country_names)
        
    #create lookup dictionary
    income_lookup = income_df.set_index('Economy')['Income group'].to_dict()
    
    # Filter to known countries only
    mask_known = df['Origin'].isin(known_countries) & df['Destination'].isin(known_countries)
    df_known = df[mask_known].copy().reset_index(drop=True)
    
    # Add income groups using vectorized map (faster than apply)
    df_known['origin_income_group'] = df_known['Origin'].map(income_lookup).fillna("Unknown")
    df_known['destination_income_group'] = df_known['Destination'].map(income_lookup).fillna("Unknown")
    
    # Convert to categorical for memory efficiency
    df_known['origin_income_group'] = pd.Categorical(df_known['origin_income_group'], categories=INCOME_ORDER)
    df_known['destination_income_group'] = pd.Categorical(df_known['destination_income_group'], categories=INCOME_ORDER)
    
    
    # Merge coordinates into main dataframe (avoids repeated lookups)
    coords_dict = countries_df.set_index('name')[['latitude', 'longitude']].to_dict('index')
    
    # Add origin coordinates
    df_known[['origin_lat', 'origin_lon']] = pd.DataFrame(
        df_known['Origin'].map(coords_dict).tolist(), index=df_known.index
    )
    
    # Add destination coordinates
    df_known[['dest_lat', 'dest_lon']] = pd.DataFrame(
        df_known['Destination'].map(coords_dict).tolist(), index=df_known.index
    )
    
    return df_known, coords_dict, income_lookup

# Load data once at startup
df_known, coords_dict, income_lookup = load_and_preprocess_data()

# Pre-calculate country list for dropdown (sorted once)
COUNTRY_LIST = sorted(set(df_known['Origin']).union(set(df_known['Destination'])))

# ============= HELPER FUNCTIONS =============

def millify(n):
    """Format large numbers with K, M, B, T suffixes"""
    if n == 0:
        return "0.0"
    
    millnames = ['', 'K', 'M', 'B', 'T']
    millidx = min(len(millnames) - 1, int(math.floor(math.log10(abs(n)) / 3)))
    scaled = n / 10**(3 * millidx)
    
    return f"{scaled:.1f}{millnames[millidx]}"



def create_trend_country(country, dff_full):
    """Create trend line chart showing both immigration and emigration"""
    
    # Calculate emigration (people leaving selected country)
    emigration_data = dff_full[dff_full['Origin'] == country]
    emigration_totals = emigration_data[ALL_YEARS].sum()
    
    # Calculate immigration (people coming to selected country)
    immigration_data = dff_full[dff_full['Destination'] == country]
    immigration_totals = immigration_data[ALL_YEARS].sum()
    
    # Create figure with both lines
    linefig = go.Figure()
    
    # Emigration line (yellow) - less prominent
    linefig.add_trace(go.Scatter(
        x=ALL_YEARS,
        y=emigration_totals,
        mode='lines+markers',
        name='Emigration',
        line=dict(width=1, color='rgba(255,193,7,0.9)'),  # Yellow
        marker=dict(size=8, color='rgba(255,193,7,0.9)')
    ))
    
    # Immigration line (white) - stands out
    linefig.add_trace(go.Scatter(
        x=ALL_YEARS,
        y=immigration_totals,
        mode='lines+markers',
        name='Immigration',
        line=dict(width=1, color='rgba(255,255,255,0.9)'),  # White
        marker=dict(size=8, color='rgba(255,255,255,0.9)')
    ))
    
    # Calculate y-axis max from both series
    y_max = max(emigration_totals.max(), immigration_totals.max())
    
    linefig.update_layout(
        height=270,
        margin=dict(l=0, r=40, t=80, b=80),
        title=dict(
            text=f'{country}:<br>migrant stock 1990-2024',
            x=0.5,  # Center horizontally
            xanchor='center'
        ),
        xaxis_title="",
        yaxis_title="",
        plot_bgcolor='#2c2f38',
        paper_bgcolor='#2c2f38',
        font=dict(color='rgba(255,255,255,0.7)'),
        legend=dict(
            # x=1.02,  # Position outside right edge
            # y=1,     # Top
            # xanchor='left',  # Anchor left edge of legend to x position
            # yanchor='top',
            orientation="h",
            yanchor="bottom",
            y=-0.60,
            xanchor="center",
            x=0.5,
  
            bgcolor='rgba(44, 47, 56, 0.8)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1
        ),
        yaxis=dict(
            range=[0, y_max * 1.1],
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.1)',
            zerolinewidth=1
        ),
        xaxis=dict(
            showgrid=False
        )
    )
    
    return dcc.Graph(figure=linefig)




def create_country_info_card(country, view,  dff):
    """Create KPI card with country information and income group distribution"""
    
    related_countries = len(dff)
    people_num = dff['2024'].sum()
    income_group_country = income_lookup.get(country, "Unknown")
    
    # Determine grouping column based on view
    groupby_col = 'origin_income_group' if view else 'destination_income_group'
    graph_title = "Profile country of origin" if view else "Profile destination country"
    
    # Group and aggregate (single operation)
    dffl = dff.groupby(groupby_col, observed=False, as_index=False)['2024'].sum()
    
    # Calculate percentages
    dffl['perc_of_people_num'] = 100 * dffl['2024'] / people_num
    dffl['percent_label'] = dffl['perc_of_people_num'].round(1).astype(str) + "%"
    
    # Ensure all income categories are present
    dffl = dffl.set_index(groupby_col).reindex(INCOME_ORDER, fill_value=0).reset_index()
    dffl['percent_label'] = dffl['percent_label'].replace('0.0%', '0%')
    
    # Add colors
    dffl['color'] = dffl[groupby_col].map(INCOME_COLORS)
    
    # Reverse for horizontal bar chart
    dffl = dffl.iloc[::-1]
    income_labels = [label + "  " for label in INCOME_ORDER[::-1]]
    
    # Create bar chart
    fig = go.Figure(go.Bar(
        x=dffl['perc_of_people_num'],
        y=dffl[groupby_col],
        orientation='h',
        text=dffl['percent_label'],
        textposition='outside',
        textfont=dict(color='white', size=12),
        marker_color=dffl['color'],
        cliponaxis=False
    ))
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        height=200,
        margin=dict(t=30, l=30, b=10, r=10),
        xaxis=dict(visible=False, range=[0, 110], fixedrange=True),
        yaxis=dict(
            showgrid=False, 
            fixedrange=True,
            tickvals=list(range(len(income_labels))),
            ticktext=income_labels,
            color="white"
        ),
        font=dict(size=10)
    )
    
    title = "Immigration" if view else "Emigration"
    
    # Create card
    card = dbc.Card([
        dbc.CardHeader([
            html.H2(f"{country}"),
            html.H5(f" Income classification: {income_group_country} *")
        ], style={"textAlign": "center"}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H3(f"{title}", className="card-title"),
                    html.H2(f"{millify(people_num)} people"),
                    html.P(
                        f"{'to' if not view else 'from'} {related_countries} countries",
                        className="card-text", style={'color': 'white'}
                    ),
                ], width=12, xl = 3, style={"textAlign": "center"}),
                dbc.Col([
                    html.H3(f"{graph_title}", className="card-title", style={"textAlign": "center"}),
                    html.Div(dcc.Graph(figure=fig))
                ], width=12, xl = 9)
            ])
        ]),
        dbc.CardFooter([
            html.P('* World Bank Classification 2024', style={"textAlign": "center","fontSize":".8rem"})
        ])
    ], style={"backgroundColor": "#66001f"})
    
    return card

# ============= UI COMPONENTS =============

select_country = html.Div([
    
    dbc.Select(
        id="country-dropdown",
        options=[{'label': c, 'value': c} for c in COUNTRY_LIST],
        placeholder="Select a country",
        value='Netherlands'
    ),
])

select_view = html.Div([
    dbc.RadioItems(
        id="view-switch",
        options=[
                {"label": "Immigration (from abroad living in)", "value": 1},
                {"label": "Emigration (living abroad)", "value": 0},
            ],
       # label="Toggle Emigration <=> Immigration",
        value=1,
        inline=True,
        persistence='session'
    ),
])

# ============= DASH APP =============

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,  vizro_bootstrap])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Migrant Stock"),
            html.P("International migration stock is the total number of foreign-born persons or foreign citizens residing in a country at a specific point in time (e.g., mid-year). The numbers belonging to the visuals are a snapshot."),
            html.P("The dataset tracks migrants by country of origin and destination. This app visualizes migration patterns based on country and income classification."),
            dbc.Row([               
                 dbc.Col(html.H3("Select a country:")),
                 dbc.Col(select_country)
             ], className="selectionRow")
        ],width=12, xl = 6),
        dbc.Col([ 
            html.Div(id='trend-chart')
           ],width=12, xl = 6
            )
    ], style={'margin': '1rem'}),
    html.Hr(className="hrGutter"),

    dbc.Row([
        dbc.Col([
            dbc.Row([               
                 dbc.Col(html.H3("View 2024 details on")),
                 dbc.Col(select_view)
             ],  className="selectionRow"),
           
            html.Div(id='country-kpi'),
                

        ],width=12, xl = 6),
        dbc.Col([
            html.H3(id="map-title", className="text-center"),
            dl.Map(center=[20, 0], zoom=2, children=[
                dl.TileLayer(),
                dl.LayerGroup(id="arrows-layer")
            ], style={'height': '60vh', 'width': '100%'}),

        ],width=12, xl = 6, style={'height': '60vh'})
    ], style={'margin': '1rem'}),
    html.Hr(className="hrGutter"),
    dbc.Row(
        dbc.Col([
            html.Div([
                html.A(
                    "Datasource: United Nations, International Migrant Stock",
                    href="https://www.un.org/development/desa/pd/content/international-migrant-stock",
                    target="_blank",
                    style={"color": "rgba(255,255,255,0.7)", "marginRight": "2rem"}
                ),
                html.Span(" | ", style={"color": "rgba(255,255,255,0.4)"}),
                html.A(
                    "Plotly Dash App: Marie-Anne Melis",
                    href="https://www.linkedin.com/in/marieannemelis/",
                    target="_blank",
                    style={"color": "rgba(255,255,255,0.7)", "marginLeft": "2rem"}
                )
            ], style={"textAlign": "center"})
        ])
    )
])

# ============= CALLBACKS =============

@app.callback(
    Output("arrows-layer", "children"),
    Output("map-title","children"),
    Output("country-kpi", "children"),
    Output("trend-chart","children"),
    Input("country-dropdown", "value"),
    Input("view-switch", "value")
)
def update_map(selected_country, selected_view):
    #Update map and KPI card based on selected country and view
    
    if not selected_country:
        selected_country = 'Netherlands'
    
    # # Calculate max y-value across BOTH immigration and emigration for consistent scaling
    immigration_data = df_known[df_known['Destination'] == selected_country]
    emigration_data = df_known[df_known['Origin'] == selected_country]
    
    # immigration_max = immigration_data[ALL_YEARS].sum().max() if len(immigration_data) > 0 else 0
    # emigration_max = emigration_data[ALL_YEARS].sum().max() if len(emigration_data) > 0 else 0
    
    # y_max = max(immigration_max, emigration_max)
    
    # Filter data based on view
    if selected_view:  # Destination view (immigration)
        dff = immigration_data.copy()
        coord_cols = ('origin_lat', 'origin_lon')
        income_col = 'origin_income_group'
        other_country_col = 'Origin'
        # ✅ Recalculate color based on current income group column
        dff['marker_color'] = dff[income_col].map(INCOME_COLORS)
        map_title = f"Immigration stock {selected_country} 2024"
    else:  # Origin view (emigration)
        dff = emigration_data.copy()
        coord_cols = ('dest_lat', 'dest_lon')
        income_col = 'destination_income_group'
        other_country_col = 'Destination'
        # ✅ Recalculate color based on current income group column
        dff['marker_color'] = dff[income_col].map(INCOME_COLORS)
        map_title = f"Emigration stock {selected_country} 2024"
    
    # Pre-calculate radius for all markers
    dff['radius'] = 3 + np.sqrt(dff['2024']) * 0.01
    
    
    
    

    # Build map elements
    map_elements = []
    
    for idx, row in dff.iterrows():
        # Add polyline
        positions = [
            [row['origin_lat'], row['origin_lon']],
            [row['dest_lat'], row['dest_lon']]
        ]
        
        map_elements.append(
            dl.Polyline(
                id=f"polyline-{selected_country}-{row[other_country_col]}-{selected_view}",  # ✅ Unique ID
                positions=positions,
                color="#ffffff",
                weight=1,
                opacity=0.2
            )
        )
        
        # Add marker for other country (not selected country)
        if row[other_country_col] != selected_country:
            map_elements.append(
                dl.CircleMarker(
                    id=f"marker-{selected_country}-{row[other_country_col]}-{selected_view}",  # ✅ Unique ID
                    center=[row[coord_cols[0]], row[coord_cols[1]]],
                    radius=row['radius'],
                    color=row['marker_color'],
                    fillColor=row['marker_color'],
                    fill=True,
                    fillOpacity=0.7,
                    children=[dl.Tooltip(
                        f"{row[other_country_col]}: {row['2024']:,} people, {row[income_col]}"
                    )]
                )
            )
    
    # Add marker for selected country
    selected_coords = coords_dict.get(selected_country)
    if selected_coords:
        map_elements.append(
            dl.CircleMarker(
                id=f"marker-selected-{selected_country}",  # ✅ Unique ID
                center=[selected_coords['latitude'], selected_coords['longitude']],
                radius=8,
                color="white",
                fillColor="white",
                fill=True,
                fillOpacity=0.7,
                children=[dl.Tooltip(f"{selected_country}")]
            )
        )
    
    return map_elements, map_title, create_country_info_card(selected_country, selected_view, dff),\
        create_trend_country(selected_country, df_known)

if __name__ == '__main__':
    app.run(debug=False)
