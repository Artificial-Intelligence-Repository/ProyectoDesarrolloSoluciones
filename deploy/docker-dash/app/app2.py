import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import requests
import json
import os

# Get API URL from environment variable or use default
api_url = os.getenv('API_URL', 'api')
api_url = f"http://{api_url}:8001/api/v1/predict"

# Implementación del modelo
def analisis_sentimiento(resena):
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    payload = {
        "inputs": [
            {
                "review": resena
            }
        ]
    }
    try:
        response = requests.post(
            api_url,
            headers=headers,
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            data = response.json()
            # La API retorna una predicción en data["predictions"][0]
            return "Positiva" if data["predictions"][0] == 1 else "Negativa"
        else:
            return f"Error en la API: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Error de conexión: {str(e)}"

# Aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(style={'backgroundColor': '#c1bbba', 'paddingLeft': '10%', 'paddingRight': '10%'}, children=[
    html.H1("CineVisión", style={'textAlign': 'center', 'color': '#000', 'backgroundColor': '#7f7c7b', 'fontSize': '32px'}),
    html.H2("Queremos conocer tu opinión", style={'textAlign': 'left', 'color': '#000', 'fontSize': '28px', 'marginBottom': '20px'}),
    html.P("Sabemos que la verdadera magia del cine no se encuentra solo en las pantallas, sino en las emociones, opiniones y experiencias que cada película despierta en cada uno de nosotros. Por eso, hemos creado una nueva aplicación y necesitamos tu ayuda para convertirla en la plataforma de reseñas de películas más auténtica y apasionante.",
           style={'textAlign': 'left', 'color': '#000', 'fontSize': '16px', 'marginBottom': '20px'}),
    html.H3("Lista de películas", style={'color': '#000', 'fontSize': '20px', 'marginBottom': '10px'}),
    dcc.Dropdown(
        id='dropdown-peliculas',
        options=[
            {'label': 'Capitán América', 'value': 'Capitan_America'},
            {'label': 'Conclave', 'value': 'Conclave'},
            {'label': 'Mufasa', 'value': 'Mufasa'},
            {'label': 'Flow', 'value': 'Flow'}
        ],
        value='Capitan_America',
        style={'width': '50%', 'fontSize': '16px', 'marginBottom': '20px'}
    ),
    html.H3("Haz una reseña acerca de la misma", style={'color': '#000', 'fontSize': '20px', 'marginBottom': '10px'}),
    dcc.Textarea(
        id='textarea-resena',
        placeholder='Escribe tu reseña aquí...',
        style={'width': '60%', 'height': 100, 'fontSize': '16px'}  # Margen adicional
    ),
    html.Div(style={'marginBottom': '3px'}),
    html.Button('Guardar reseña', id='boton-guardar', n_clicks=0, style={'width': '15%', 'fontSize': '16px', 'marginBottom': '20px'}),
    html.H3("Resultado del análisis de sentimiento:", style={'color': '#000', 'fontSize': '20px', 'marginBottom': '10px'}),
    html.Div(id='resultado-analisis', style={'color': '#000', 'fontWeight': 'bold', 'fontSize': '20px', 'marginBottom': '20px'})
])

@app.callback(
    Output('resultado-analisis', 'children'),
    [Input('boton-guardar', 'n_clicks')],
    [State('textarea-resena', 'value')]
)
def actualizar_resultado(n_clicks, resena):
    if n_clicks > 0 and resena:
        resultado = analisis_sentimiento(resena)
        return f'La reseña es {resultado}'
    return ''

if __name__ == '__main__':
    # app.run_server(host ="0.0.0.0", debug=True)
    app.run_server(
        host="0.0.0.0",
        port=8000,
        debug=True,  # Enable debug mode
        use_reloader=True  # Enable hot reloading
    )

