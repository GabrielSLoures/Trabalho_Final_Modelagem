"""
TRABALHO FINAL - MODELAGEM E PROGRAMAÇÃO ESTATÍSTICA
Tema: Saúde Mental no Brasil - Análise da Série "Saúde Mental em Dados"

Dashboard interativo usando Plotly Dash
Autores: Gabriel Loures e Kauã Fernandes
Data: Outubro 2025
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from pathlib import Path
from functools import lru_cache
import os

# ============================================================================
# CONFIGURAÇÃO DOS DADOS
# ============================================================================

DATA_DIR = Path(__file__).parent


def _read_or_create_csv(name: str, df_default: pd.DataFrame):
    """(DEPRECADO) função antiga - não usada. Mantida por compatibilidade.

    Use _read_csv_required em vez desta função.
    """
    path = DATA_DIR / name
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError(f"Arquivo de dados esperado não encontrado: {path}\nPor favor, coloque o CSV correspondente em {DATA_DIR}.")


# Ler CSVs obrigatórios (sem dados embutidos)
def _read_csv_required(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de dados esperado não encontrado: {path}\nPor favor, coloque os CSVs 'caps.csv', 'leitos.csv', 'financiamento.csv', 'regiao.csv' e 'pvc.csv' em {DATA_DIR}"
        )
    return pd.read_csv(path)


# Carrega os datasets a partir dos CSVs
df_caps = _read_csv_required('caps.csv')
df_leitos = _read_csv_required('leitos.csv')
df_financiamento = _read_csv_required('financiamento.csv')
df_financiamento['Gasto_Extra_Milhoes'] = df_financiamento['Gasto_Total_Milhoes'] * df_financiamento['Perc_Extra_Hospitalar'] / 100
df_financiamento['Gasto_Hosp_Milhoes'] = df_financiamento['Gasto_Total_Milhoes'] * df_financiamento['Perc_Hospitalar'] / 100
df_regiao = _read_csv_required('regiao.csv')
ano_ultimo = df_regiao['Ano'].max()
df_ultimo = df_regiao[df_regiao['Ano'] == ano_ultimo]
ranking_regioes = df_ultimo.sort_values('Cobertura', ascending=False)
desvio_padrao = df_ultimo['Cobertura'].std()
coef_var = desvio_padrao / df_ultimo['Cobertura'].mean()

# Mapa de cores consistente por região (usado em múltiplos gráficos)
# Ordenamento determinístico: ordena alfabeticamente as regiões e associa cores
_REGIOES_ORDERED = sorted(df_regiao['Regiao'].unique())
_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
# Garantir que existam cores suficientes
if len(_PALETTE) < len(_REGIOES_ORDERED):
    # repetir palette se necessário
    from itertools import cycle, islice
    _PALETTE = list(islice(cycle(_PALETTE), len(_REGIOES_ORDERED)))
COLOR_MAP = {reg: col for reg, col in zip(_REGIOES_ORDERED, _PALETTE)}
df_pvc = _read_csv_required('pvc.csv')

# Taxa de crescimento anual beneficiários
taxa_benef_ano = ((df_pvc['Beneficiarios'].iloc[-1]/df_pvc['Beneficiarios'].iloc[0])**(1/(df_pvc.shape[0]-1))-1)*100
# Taxa de crescimento anual valor
taxa_valor_ano = ((df_pvc['Valor_Mensal_RS'].iloc[-1]/df_pvc['Valor_Mensal_RS'].iloc[0])**(1/(df_pvc.shape[0]-1))-1)*100

# Projeção linear beneficiários (cached)
@lru_cache(maxsize=8)
def _fit_modelo_pvc():
    from sklearn.linear_model import LinearRegression
    X_pvc = df_pvc['Ano'].values.reshape(-1, 1)
    y_pvc = df_pvc['Beneficiarios'].values
    modelo = LinearRegression()
    modelo.fit(X_pvc, y_pvc)
    return modelo

modelo_pvc = _fit_modelo_pvc()
proj_benef_2026 = int(modelo_pvc.predict(np.array([[2026]]))[0])
proj_benef_2030 = int(modelo_pvc.predict(np.array([[2030]]))[0])

# ============================================================================
# MODELAGEM ESTATÍSTICA
# ============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.stats import t

# Modelo CAPS

# Modelo CAPS
X_caps = df_caps['Ano'].values.reshape(-1, 1)
y_caps = df_caps['Total_CAPS'].values
modelo_caps = LinearRegression()
modelo_caps.fit(X_caps, y_caps)
r2_caps = modelo_caps.score(X_caps, y_caps)

# Projeção com intervalo de confiança (95%)
def projecao_caps_intervalo(anos):
    """Retorna (previsoes, intervalos) para array de anos.

    Aceita `anos` como array 1D ([2026,2027]) ou 2D (shape (-1,1)).
    """
    anos_arr = np.asarray(anos)
    # padroniza shape para (n,1) para predict
    if anos_arr.ndim == 1:
        anos_reshaped = anos_arr.reshape(-1, 1)
    else:
        anos_reshaped = anos_arr

    pred = modelo_caps.predict(anos_reshaped)
    n = len(X_caps)
    mean_x = np.mean(X_caps.ravel())
    t_value = t.ppf(0.975, df=n-2)
    se = np.sqrt(np.sum((y_caps - modelo_caps.predict(X_caps))**2) / (n-2))
    sx = np.sum((X_caps.ravel() - mean_x)**2)
    intervalos = []
    for x in anos_reshaped.ravel():
        erro = se * np.sqrt(1/n + ((x-mean_x)**2)/sx)
        intervalo = t_value * erro
        intervalos.append(intervalo)
    return pred, np.array(intervalos)

# Modelo Leitos
X_leitos = df_leitos['Ano'].values.reshape(-1, 1)
y_leitos = df_leitos['Leitos_Psiquiatricos'].values
modelo_leitos = LinearRegression()
modelo_leitos.fit(X_leitos, y_leitos)
r2_leitos = modelo_leitos.score(X_leitos, y_leitos)

# Correlação CAPS-Leitos

# Correlação CAPS-Leitos
from scipy.stats import pearsonr
correlacao, pval_correlacao = pearsonr(df_caps['Total_CAPS'], df_leitos['Leitos_Psiquiatricos'])

# Projeções 2026-2030
anos_futuros = np.array([2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
caps_projetados = modelo_caps.predict(anos_futuros)

# ============================================================================
# INICIALIZAR APLICAÇÃO DASH
# ============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard - Saúde Mental no Brasil"

# ============================================================================
# FUNÇÕES PARA CRIAR GRÁFICOS
# ============================================================================

def criar_grafico_evolucao_caps():
    """Gráfico de linha: evolução dos CAPS"""
    fig = go.Figure()
    # Linha principal
    fig.add_trace(go.Scatter(
        x=df_caps['Ano'],
        y=df_caps['Total_CAPS'],
        mode='lines+markers',
        name='CAPS',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    # Linha de tendência
    anos_trend = df_caps['Ano'].values.reshape(-1, 1)
    caps_trend = modelo_caps.predict(anos_trend)
    fig.add_trace(go.Scatter(
        x=df_caps['Ano'],
        y=caps_trend,
        mode='lines',
        name='Tendência Linear',
        line=dict(color='red', width=2, dash='dash')
    ))
    # Projeção com intervalo de confiança
    anos_futuros = np.array([2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
    proj, intervalo = projecao_caps_intervalo(anos_futuros)
    fig.add_trace(go.Scatter(
        x=anos_futuros.flatten(),
        y=proj,
        mode='lines+markers',
        name='Projeção',
        line=dict(color='purple', width=2, dash='dot'),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=anos_futuros.flatten(),
        y=proj+intervalo,
        mode='lines',
        name='Limite Superior (95%)',
        line=dict(color='rgba(128,0,128,0.2)', width=1, dash='dot'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=anos_futuros.flatten(),
        y=proj-intervalo,
        mode='lines',
        name='Limite Inferior (95%)',
        line=dict(color='rgba(128,0,128,0.2)', width=1, dash='dot'),
        fill='tonexty',
        showlegend=False
    ))
    fig.update_layout(
        title='Evolução e Projeção do Número de CAPS no Brasil (2002-2030)',
        xaxis_title='Ano',
        yaxis_title='Número de CAPS',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

def criar_grafico_tipos_caps():
    """Gráfico de barras empilhadas: tipos de CAPS"""
    anos_selecionados = df_caps[df_caps['Ano'].isin([2002, 2006, 2010, 2015, 2024])]
    
    fig = go.Figure()
    
    tipos = ['CAPS_I', 'CAPS_II', 'CAPS_III', 'CAPSi', 'CAPSad']
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for tipo, cor in zip(tipos, cores):
        fig.add_trace(go.Bar(
            name=tipo.replace('_', ' '),
            x=anos_selecionados['Ano'],
            y=anos_selecionados[tipo],
            marker_color=cor
        ))
    
    fig.update_layout(
        title='Distribuição dos CAPS por Tipo (Anos Selecionados)',
        xaxis_title='Ano',
        yaxis_title='Quantidade',
        barmode='stack',
        template='plotly_white'
    )
    
    return fig

def criar_grafico_caps_vs_leitos():
    """Gráfico com dois eixos Y: CAPS vs Leitos"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # CAPS (eixo esquerdo)
    fig.add_trace(
        go.Scatter(
            x=df_caps['Ano'],
            y=df_caps['Total_CAPS'],
            name='CAPS',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ),
        secondary_y=False
    )
    # Leitos (eixo direito)
    fig.add_trace(
        go.Scatter(
            x=df_leitos['Ano'],
            y=df_leitos['Leitos_Psiquiatricos'],
            name='Leitos Psiquiátricos',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    fig.update_xaxes(title_text="Ano")
    fig.update_yaxes(title_text="Número de CAPS", secondary_y=False)
    fig.update_yaxes(title_text="Leitos Psiquiátricos", secondary_y=True)
    fig.update_layout(
        title='Reforma Psiquiátrica: CAPS vs Leitos Hospitalares',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

# Gráfico de dispersão CAPS vs Leitos
def grafico_dispersa_caps_leitos():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_caps['Total_CAPS'],
        y=df_leitos['Leitos_Psiquiatricos'],
        mode='markers+text',
        text=df_caps['Ano'],
        textposition='top center',
        marker=dict(color='blue', size=10),
        name='Ano'
    ))
    # Linha de tendência
    modelo = LinearRegression()
    X = df_caps['Total_CAPS'].values.reshape(-1, 1)
    y = df_leitos['Leitos_Psiquiatricos'].values
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    fig.add_trace(go.Scatter(
        x=df_caps['Total_CAPS'],
        y=y_pred,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Tendência'
    ))
    fig.update_layout(
        title='Dispersão: CAPS vs Leitos Psiquiátricos',
        xaxis_title='Total de CAPS',
        yaxis_title='Leitos Psiquiátricos',
        template='plotly_white'
    )
    return fig

def criar_grafico_financiamento():
    """Gráfico de área: inversão do financiamento"""
    fig = go.Figure()
    # Percentuais
    fig.add_trace(go.Scatter(
        x=df_financiamento['Ano'],
        y=df_financiamento['Perc_Extra_Hospitalar'],
        fill='tonexty',
        name='Extra-Hospitalar (%)',
        line=dict(color='#2ca02c'),
        fillcolor='rgba(44, 160, 44, 0.3)'
    ))
    fig.add_trace(go.Scatter(
        x=df_financiamento['Ano'],
        y=df_financiamento['Perc_Hospitalar'],
        fill='tozeroy',
        name='Hospitalar (%)',
        line=dict(color='#d62728'),
        fillcolor='rgba(214, 39, 40, 0.3)'
    ))
    # Linha de 50%
    fig.add_hline(y=50, line_dash="dash", line_color="black", annotation_text="Inversão (2006)")
    fig.update_layout(
        title='Inversão do Modelo de Financiamento em Saúde Mental',
        xaxis_title='Ano',
        yaxis_title='Percentual (%)',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

# Gráfico de barras: valores absolutos
def grafico_barras_financiamento():
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_financiamento['Ano'],
        y=df_financiamento['Gasto_Extra_Milhoes'],
        name='Extra-Hospitalar',
        marker_color='#2ca02c'
    ))
    fig.add_trace(go.Bar(
        x=df_financiamento['Ano'],
        y=df_financiamento['Gasto_Hosp_Milhoes'],
        name='Hospitalar',
        marker_color='#d62728'
    ))
    fig.update_layout(
        barmode='group',
        title='Gastos Absolutos em Saúde Mental (Milhões R$)',
        xaxis_title='Ano',
        yaxis_title='Milhões de R$',
        template='plotly_white'
    )
    return fig

def criar_grafico_cobertura_regional(df=None):
    """Gráfico de linhas: cobertura por região.

    Usa `df` interno; se quiser filtrar antes, passe um DataFrame filtrado como parâmetro.
    """
    # se df não fornecido, usa o global
    if df is None:
        df = df_regiao
    fig = go.Figure()
    if df is None or df.empty:
        return fig
    regioes = df['Regiao'].unique()
    for regiao in regioes:
        dados_regiao = df[df['Regiao'] == regiao]
        cor = COLOR_MAP.get(regiao, '#888')
        fig.add_trace(go.Scatter(
            x=dados_regiao['Ano'],
            y=dados_regiao['Cobertura'],
            mode='lines+markers',
            name=regiao,
            line=dict(color=cor, width=2),
            marker=dict(size=8),
            hovertemplate='%{x}: %{y:.2f} CAPS/100k<extra>%{fullData.name}</extra>'
        ))
    # Linha de cobertura "muito boa" (0.70)
    fig.add_hline(y=0.70, line_dash="dash", line_color="gray", annotation_text="Cobertura Muito Boa (0.70)")
    fig.update_layout(
        title='Evolução da Cobertura de CAPS por Região',
        xaxis_title='Ano',
        yaxis_title='Cobertura (CAPS/100k hab)',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

# Gráfico de barras do ranking das regiões
def grafico_ranking_regioes():
    fig = go.Figure()
    colors = [COLOR_MAP.get(r, '#888') for r in ranking_regioes['Regiao']]
    fig.add_trace(go.Bar(
        x=ranking_regioes['Regiao'],
        y=ranking_regioes['Cobertura'],
        marker_color=colors,
        text=[f"{v:.2f}" for v in ranking_regioes['Cobertura']],
        textposition='auto',
        name='Cobertura'
    ))
    fig.update_layout(
        title=f'Ranking de Cobertura Regional ({ano_ultimo})',
        xaxis_title='Região',
        yaxis_title='Cobertura (CAPS/100k hab)',
        template='plotly_white'
    )
    return fig

def criar_grafico_pvc():
    """Gráfico combinado: Programa De Volta para Casa"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Beneficiários (barras)
    fig.add_trace(
        go.Bar(
            x=df_pvc['Ano'],
            y=df_pvc['Beneficiarios'],
            name='Beneficiários',
            marker_color='#1f77b4'
        ),
        secondary_y=False
    )
    # Valor mensal (linha)
    fig.add_trace(
        go.Scatter(
            x=df_pvc['Ano'],
            y=df_pvc['Valor_Mensal_RS'],
            name='Valor Mensal (R$)',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    fig.update_xaxes(title_text="Ano")
    fig.update_yaxes(title_text="Beneficiários", secondary_y=False)
    fig.update_yaxes(title_text="Valor Mensal (R$)", secondary_y=True)
    fig.update_layout(
        title='Programa De Volta para Casa: Beneficiários e Valor do Auxílio',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

# Gráfico de dispersão Beneficiários vs Valor
def grafico_dispersa_pvc():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_pvc['Beneficiarios'],
        y=df_pvc['Valor_Mensal_RS'],
        mode='markers+text',
        text=df_pvc['Ano'],
        textposition='top center',
        marker=dict(color='purple', size=10),
        name='Ano'
    ))
    fig.update_layout(
        title='Dispersão: Beneficiários vs Valor do Auxílio',
        xaxis_title='Beneficiários',
        yaxis_title='Valor Mensal (R$)',
        template='plotly_white'
    )
    return fig

# ============================================================================
# LAYOUT DO DASHBOARD
# ============================================================================

app.layout = dbc.Container([
    # HEADER
    dbc.Row([
        dbc.Col([
            html.H1("Saúde Mental no Brasil - Dashboard BI", 
                    className="text-center mb-3",
                    style={
                        'background': 'linear-gradient(90deg, #4e54c8 0%, #8f94fb 100%)',
                        'color': 'white',
                        'padding': '28px',
                        'borderRadius': '18px',
                        'boxShadow': '0 4px 16px rgba(78,84,200,0.15)',
                        'fontWeight': 'bold',
                        'letterSpacing': '1px'
                    }),
            html.H4("Série 'Saúde Mental em Dados' - Ministério da Saúde (2002-2024)",
                    className="text-center mb-4",
                    style={
                        'color': '#4e54c8',
                        'fontWeight': '500',
                        'letterSpacing': '0.5px'
                    })
        ])
    ]),
    
    # KPIs PRINCIPAIS
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("3.019", className="mb-1", style={"color": "#4e54c8", "fontWeight": "bold"}),
                    html.P("CAPS em 2024", className="mb-1", style={"color": "#222", "fontSize": "1.1em"}),
                    html.Small("+612% desde 2002", className="text-success", style={"fontWeight": "500"})
                ])
            ], className="text-center shadow", style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(78,84,200,0.10)", "background": "linear-gradient(90deg, #e0eafc 0%, #cfdef3 100%)"})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("1,13", className="mb-1", style={"color": "#4e54c8", "fontWeight": "bold"}),
                    html.P("CAPS/100k hab", className="mb-1", style={"color": "#222", "fontSize": "1.1em"}),
                    html.Small("Cobertura 2024", className="text-success", style={"fontWeight": "500"})
                ])
            ], className="text-center shadow", style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(78,84,200,0.10)", "background": "linear-gradient(90deg, #e0eafc 0%, #cfdef3 100%)"})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("-29.393", className="mb-1", style={"color": "#d62728", "fontWeight": "bold"}),
                    html.P("Leitos Fechados", className="mb-1", style={"color": "#222", "fontSize": "1.1em"}),
                    html.Small("Redução de 57%", style={"color": "#b36b00", "fontWeight": "500"})
                ])
            ], className="text-center shadow", style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(214,39,40,0.10)", "background": "linear-gradient(90deg, #f8e0e0 0%, #f3cfde 100%)"})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("79,5%", className="mb-1", style={"color": "#2ca02c", "fontWeight": "bold"}),
                    html.P("Extra-Hospitalar", className="mb-1", style={"color": "#222", "fontSize": "1.1em"}),
                    html.Small("Inversão em 2006", className="text-info", style={"fontWeight": "500"})
                ])
            ], className="text-center shadow", style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(44,160,44,0.10)", "background": "linear-gradient(90deg, #eafce0 0%, #def3cf 100%)"})
        ], width=3),
    ], className="mb-4"),
    
    # TABS
    dbc.Tabs([
        # NOVA ABA: VISÃO GERAL
        dbc.Tab(label="Visão Geral", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2("Saúde Mental no Brasil: Visão Geral", className="text-center text-primary mb-3"),
                            html.Hr(),
                            html.P([
                                "Este dashboard apresenta uma análise estatística e visual dos avanços da saúde mental no Brasil, com base na série oficial 'Saúde Mental em Dados' do Ministério da Saúde (2002-2025).",
                                html.Br(),
                                html.Br(),
                                html.Strong("Objetivos:"),
                                html.Ul([
                                    html.Li("Mostrar a evolução dos serviços comunitários (CAPS) e a reforma psiquiátrica."),
                                    html.Li("Analisar a desinstitucionalização e a redução dos leitos hospitalares."),
                                    html.Li("Explorar o financiamento e a cobertura regional dos serviços."),
                                    html.Li("Apresentar o impacto do Programa De Volta para Casa."),
                                    html.Li("Aplicar modelagem estatística para projeções e correlações.")
                                ]),
                                html.Br(),
                                html.Strong("Fontes dos dados:"),
                                " Série 'Saúde Mental em Dados' (Edições 1 a 13), Ministério da Saúde. Últimos dados disponíveis: fevereiro de 2025.",
                                html.Br(),
                                html.Strong("Estrutura do dashboard:"),
                                html.Ul([
                                    html.Li("Evolução dos CAPS"),
                                    html.Li("Desinstitucionalização"),
                                    html.Li("Financiamento"),
                                    html.Li("Cobertura Regional"),
                                    html.Li("Programa De Volta para Casa"),
                                    html.Li("Modelagem Estatística")
                                ]),
                                html.Br(),
                                html.Strong("Observação importante:"),
                                " Há um gap de dados entre 2015 e 2025 devido à ausência de relatórios oficiais nesse período."
                            ], className="mb-2", style={"fontSize": "1.15em"})
                        ])
                    ], className="shadow", style={"borderRadius": "18px", "background": "linear-gradient(90deg, #e0eafc 0%, #cfdef3 100%)", "boxShadow": "0 2px 8px rgba(78,84,200,0.10)", "padding": "18px"})
                ], width=10)
            ], className="mt-4 justify-content-center")
        ]),
        # ABA 1: EVOLUÇÃO CAPS
        dbc.Tab(label="Evolução dos CAPS", children=[
            dbc.Row([
                dbc.Col([
                    html.Div(
                        dcc.Graph(figure=criar_grafico_evolucao_caps()),
                        style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(78,84,200,0.10)", "border": "2px solid #e0eafc", "padding": "8px", "background": "#fff"}
                    )
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Análise Detalhada", className="text-primary"),
                            html.Hr(),
                            html.P([
                                html.Strong("Modelo Linear:"),
                                html.Br(),
                                f"R² = {r2_caps:.4f}",
                                html.Br(),
                                f"Taxa média: {float(modelo_caps.coef_[0]):.2f} CAPS/ano",
                                html.Br(),
                                f"Crescimento total: {((df_caps['Total_CAPS'].iloc[-1]/df_caps['Total_CAPS'].iloc[0])-1)*100:.1f}% (2002-2024)",
                                html.Br(),
                                f"Ano de maior aceleração: {df_caps['Ano'][np.argmax(np.diff(df_caps['Total_CAPS']))+1]} (+{np.max(np.diff(df_caps['Total_CAPS']))} CAPS)",
                            ]),
                            html.Hr(),
                            html.P([
                                html.Strong("Projeções com Intervalo de Confiança (95%):"),
                                html.Br(),
                                f"2026: {int(modelo_caps.predict(np.array([[2026]]))[0])} CAPS ± {int(projecao_caps_intervalo(np.array([[2026]]))[1][0])}",
                                html.Br(),
                                f"2030: {int(modelo_caps.predict(np.array([[2030]]))[0])} CAPS ± {int(projecao_caps_intervalo(np.array([[2030]]))[1][0])}",
                            ]),
                            html.Hr(),
                            html.P([
                                html.Strong("Comentário Analítico:"),
                                html.Br(),
                                "O crescimento dos CAPS é consistente e acelerado, com destaque para o período de 2006 a 2008. As projeções indicam manutenção da tendência positiva, com intervalo de confiança estatístico."
                            ])
                        ])
                    ], className="shadow")
                ], width=4)
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    html.Div(
                        dcc.Graph(figure=criar_grafico_tipos_caps()),
                        style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(78,84,200,0.10)", "border": "2px solid #e0eafc", "padding": "8px", "background": "#fff"}
                    )
                ], width=12)
            ], className="mt-3")
        ]),
        
        # ABA 2: DESINSTITUCIONALIZAÇÃO
        dbc.Tab(label="Desinstitucionalização", children=[
            dbc.Row([
                dbc.Col([
                    html.Div(
                        dcc.Graph(figure=criar_grafico_caps_vs_leitos()),
                        style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(78,84,200,0.10)", "border": "2px solid #e0eafc", "padding": "8px", "background": "#fff"}
                    )
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Análise Detalhada da Correlação", className="text-primary"),
                            html.Hr(),
                            html.P([
                                html.Strong("Coeficiente de Pearson:"),
                                f" {correlacao:.4f}",
                                html.Br(),
                                html.Strong("Valor-p:"),
                                f" {pval_correlacao:.2e}",
                                html.Br(),
                                "Correlação negativa forte e altamente significativa (p < 0.001)",
                                html.Br(),
                                "Indica que o aumento dos CAPS está fortemente associado à redução dos leitos psiquiátricos, evidenciando a substituição do modelo manicomial pelo comunitário."
                            ]),
                            html.Hr(),
                            html.P([
                                html.Strong("Dispersão CAPS vs Leitos:"),
                                html.Br(),
                                "Veja abaixo a relação direta entre o número de CAPS e a quantidade de leitos psiquiátricos ao longo dos anos."
                            ])
                        ])
                    ], className="shadow")
                ], width=4)
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    html.Div(
                        dcc.Graph(figure=grafico_dispersa_caps_leitos()),
                        style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(78,84,200,0.10)", "border": "2px solid #e0eafc", "padding": "8px", "background": "#fff"}
                    )
                ], width=12)
            ], className="mt-3")
        ]),
        
        # ABA 3: FINANCIAMENTO
        dbc.Tab(label="Financiamento", children=[
            dbc.Row([
                dbc.Col([
                    html.Div(
                        dcc.Graph(figure=criar_grafico_financiamento()),
                        style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(78,84,200,0.10)", "border": "2px solid #e0eafc", "padding": "8px", "background": "#fff"}
                    )
                ], width=6),
                dbc.Col([
                    html.Div(
                        dcc.Graph(figure=grafico_barras_financiamento()),
                        style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(78,84,200,0.10)", "border": "2px solid #e0eafc", "padding": "8px", "background": "#fff"}
                    )
                ], width=6)
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Análise Detalhada do Financiamento", className="text-primary"),
                            html.Hr(),
                            html.P([
                                html.Strong("Crescimento acumulado extra-hospitalar:"),
                                f" {((df_financiamento['Gasto_Extra_Milhoes'].iloc[-1]/df_financiamento['Gasto_Extra_Milhoes'].iloc[0])-1)*100:.1f}% (2002-2015)",
                                html.Br(),
                                html.Strong("Crescimento acumulado hospitalar:"),
                                f" {((df_financiamento['Gasto_Hosp_Milhoes'].iloc[-1]/df_financiamento['Gasto_Hosp_Milhoes'].iloc[0])-1)*100:.1f}% (2002-2015)",
                                html.Br(),
                                html.Strong("Ano de maior variação extra-hospitalar:"),
                                f" {df_financiamento['Ano'][np.argmax(np.diff(df_financiamento['Gasto_Extra_Milhoes']))+1]} (+{np.max(np.diff(df_financiamento['Gasto_Extra_Milhoes'])):.1f} milhões)",
                                html.Br(),
                                html.Strong("Ano de maior variação hospitalar:"),
                                f" {df_financiamento['Ano'][np.argmax(np.diff(df_financiamento['Gasto_Hosp_Milhoes']))+1]} ({np.max(np.diff(df_financiamento['Gasto_Hosp_Milhoes'])):.1f} milhões)",
                            ]),
                            html.Hr(),
                            html.P([
                                html.Strong("Comentário Analítico:"),
                                html.Br(),
                                "A inversão histórica do financiamento em 2006 marca a transição do modelo hospitalar para o comunitário, com crescimento expressivo dos gastos extra-hospitalares e redução dos hospitalares."
                            ])
                        ])
                    ], className="shadow")
                ], width=12)
            ], className="mt-3")
        ]),
        
        # ABA 4: COBERTURA REGIONAL
        dbc.Tab(label="Cobertura Regional", children=[
            dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Label("Selecione região(es):"),
                                    dcc.Dropdown(
                                        id='region-select',
                                        options=[{'label': r, 'value': r} for r in sorted(df_regiao['Regiao'].unique())],
                                        value=list(sorted(df_regiao['Regiao'].unique())),
                                        multi=True,
                                        placeholder='Todas as regiões'
                                    ),
                                    html.Br(),
                                    html.Label("Intervalo de anos:"),
                                    dcc.RangeSlider(
                                        id='year-range',
                                        min=int(df_regiao['Ano'].min()),
                                        max=int(df_regiao['Ano'].max()),
                                        step=1,
                                        value=[int(df_regiao['Ano'].min()), int(df_regiao['Ano'].max())],
                                        marks={int(y): str(int(y)) for y in sorted(df_regiao['Ano'].unique())}
                                    ),
                                    html.Br(),
                                    dbc.Button("Download CSV (filtrado)", id='btn-download', color='primary', className='mb-2'),
                                    dcc.Download(id='download-regiao'),
                                    html.Div(
                                        dcc.Graph(id='cobertura-fig', figure=criar_grafico_cobertura_regional(df_regiao)),
                                        style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(78,84,200,0.10)", "border": "2px solid #e0eafc", "padding": "8px", "background": "#fff"}
                                    )
                                ])
                            ], className='shadow')
                        ], width=8),
                dbc.Col([
                            html.Div(
                                dcc.Graph(id='ranking-fig', figure=grafico_ranking_regioes()),
                                style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(78,84,200,0.10)", "border": "2px solid #e0eafc", "padding": "8px", "background": "#fff"}
                            )
                ], width=4)
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Análise Detalhada Regional", className="text-primary"),
                            html.Hr(),
                            html.P([
                                html.Strong("Região com maior cobertura:"),
                                f" {ranking_regioes['Regiao'].iloc[0]} ({ranking_regioes['Cobertura'].iloc[0]:.2f})",
                                html.Br(),
                                html.Strong("Região com menor cobertura:"),
                                f" {ranking_regioes['Regiao'].iloc[-1]} ({ranking_regioes['Cobertura'].iloc[-1]:.2f})",
                                html.Br(),
                                html.Strong("Desvio padrão entre regiões:"),
                                f" {desvio_padrao:.2f}",
                                html.Br(),
                                html.Strong("Coeficiente de variação:"),
                                f" {coef_var:.2f}",
                            ]),
                            html.Hr(),
                            html.P([
                                html.Strong("Comentário Analítico:"),
                                html.Br(),
                                "Apesar do avanço nacional, há desigualdade regional relevante na cobertura de CAPS, com o Sul liderando e o Norte/Centro-Oeste apresentando menor cobertura."
                            ])
                        ])
                    ], className="shadow")
                ], width=12)
            ], className="mt-3")
        ]),
        
        # ABA 5: PROGRAMA DE VOLTA PARA CASA
        dbc.Tab(label="Programa De Volta para Casa", children=[
            dbc.Row([
                dbc.Col([
                    html.Div(
                        dcc.Graph(figure=criar_grafico_pvc()),
                        style={"borderRadius": "16px", "boxShadow": "0 2px 8px rgba(78,84,200,0.10)", "border": "2px solid #e0eafc", "padding": "8px", "background": "#fff"}
                    )
                ], width=12)
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Análise Avançada do Programa", className="text-primary"),
                            html.Hr(),
                            html.P([
                                html.Strong("Expansão dos beneficiários:"),
                                f" O número de beneficiários cresceu de 206 para 4.800 entre 2003 e 2024, um aumento de mais de 2.200%.",
                                html.Br(),
                                html.Strong("Crescimento anual médio:"),
                                f" {taxa_benef_ano:.1f}% ao ano, mostrando forte adesão e impacto social.",
                                html.Br(),
                                html.Strong("Evolução do valor do auxílio:"),
                                f" O valor mensal subiu de R$ 240 para R$ 622 (+159%), acompanhando a valorização do programa.",
                                html.Br(),
                                html.Strong("Projeções futuras:"),
                                f" Mantida a tendência, estima-se cerca de {proj_benef_2026} beneficiários em 2026 e {proj_benef_2030} em 2030.",
                                html.Br(),
                                html.Strong("Impacto social:"),
                                " O programa é fundamental para a reinserção social de pessoas egressas de hospitais psiquiátricos, promovendo cidadania e autonomia."
                            ])
                        ])
                    ], className="shadow")
                ], width=12)
            ], className="mt-3")
        ]),
        
        # ABA 6: MODELAGEM
        dbc.Tab(label="Modelagem Estatística", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Análise Avançada dos Modelos Estatísticos", className="text-center text-primary"),
                            html.Hr(),
                            html.H6("1. Crescimento dos CAPS"),
                            html.P([
                                f"O modelo linear apresenta R² = {r2_caps:.4f}, indicando excelente ajuste. A taxa média de crescimento é de {float(modelo_caps.coef_[0]):.2f} CAPS/ano, evidenciando expansão acelerada e sustentada do serviço comunitário.",
                                html.Br(),
                                "A projeção para 2030 sugere continuidade da tendência, com intervalo de confiança estatístico."
                            ]),
                            html.H6("2. Redução dos Leitos Psiquiátricos"),
                            html.P([
                                f"O modelo linear para leitos apresenta R² = {r2_leitos:.4f}, mostrando ajuste robusto. A taxa de redução é de {abs(float(modelo_leitos.coef_[0])):.2f} leitos/ano, refletindo o avanço da reforma psiquiátrica.",
                                html.Br(),
                                "A diminuição dos leitos é consistente e acompanha o crescimento dos CAPS."
                            ]),
                            html.H6("3. Correlação CAPS-Leitos"),
                            html.P([
                                f"O coeficiente de Pearson é {correlacao:.4f} (p < 0.001), indicando correlação negativa forte e estatisticamente significativa.",
                                html.Br(),
                                "Isso confirma que o aumento dos CAPS está diretamente associado à redução dos leitos hospitalares, evidenciando a substituição do modelo manicomial pelo comunitário."
                            ]),
                            html.Hr(),
                            html.H5("Interpretação e Implicações", className="text-success"),
                            html.P([
                                "Os resultados estatísticos confirmam a transformação estrutural da atenção em saúde mental no Brasil. O excelente ajuste dos modelos e a alta significância reforçam a efetividade das políticas públicas de desinstitucionalização e expansão dos serviços comunitários.",
                                html.Br(),
                                "Limitações: Modelos lineares não capturam possíveis efeitos não-lineares ou fatores externos. Recomenda-se monitoramento contínuo e análise complementar para decisões futuras."
                            ])
                        ])
                    ], className="shadow")
                ], width=12)
            ], className="mt-3")
        ])
    ]),
    
    # FOOTER
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.P([
                "Fonte: Ministério da Saúde - Série 'Saúde Mental em Dados' (Edições 1-13) | ",
                "Período: 2002-2024 | ",
                "Trabalho: Modelagem e Programação Estatística"
            ], className="text-center text-muted small", style={"marginBottom": "0px"}),
            html.P([
                html.Strong("Observação sobre os dados: "),
                "Entre outubro de 2015 (12º relatório) e fevereiro de 2025 (13º relatório), não há publicações oficiais da série 'Saúde Mental em Dados'. Por isso, existe um gap de informações nesse período, e os dados de 2024/2025 refletem apenas o último relatório disponível."
            ], className="text-center text-warning small", style={"marginTop": "10px", "background": "#fffbe6", "borderRadius": "10px", "padding": "8px", "boxShadow": "0 1px 4px rgba(255,193,7,0.10)"})
        ])
    ], style={"marginBottom": "-10px"})
    
], fluid=True, style={'padding': '20px'})

# ============================================================================
# EXECUTAR APLICAÇÃO
# ============================================================================


# Callbacks: atualizar gráfico regional (filtros e download)
@app.callback(
    Output('cobertura-fig', 'figure'),
    Output('ranking-fig', 'figure'),
    Input('region-select', 'value'),
    Input('year-range', 'value')
)
def update_regional(selected_regions, year_range):
    # validações simples
    if selected_regions is None or len(selected_regions) == 0:
        selected_regions = list(df_regiao['Regiao'].unique())
    yr0, yr1 = year_range
    df_f = df_regiao[(df_regiao['Regiao'].isin(selected_regions)) & (df_regiao['Ano'] >= yr0) & (df_regiao['Ano'] <= yr1)]
    fig1 = criar_grafico_cobertura_regional(df_f)

    # ranking pela maior cobertura no intervalo selecionado (último ano disponível)
    if not df_f.empty:
        ano_local = int(df_f['Ano'].max())
        df_last = df_f[df_f['Ano'] == ano_local].sort_values('Cobertura', ascending=False)
    else:
        df_last = pd.DataFrame(columns=['Regiao', 'Cobertura'])

    fig2 = go.Figure()
    if not df_last.empty:
        colors = [COLOR_MAP.get(r, '#888') for r in df_last['Regiao']]
        fig2.add_trace(go.Bar(
            x=df_last['Regiao'],
            y=df_last['Cobertura'],
            marker_color=colors,
            text=[f"{v:.2f}" for v in df_last['Cobertura']],
            textposition='auto'
        ))
        fig2.update_layout(
            title=f'Ranking de Cobertura Regional ({ano_local})',
            xaxis_title='Região',
            yaxis_title='Cobertura (CAPS/100k hab)',
            template='plotly_white'
        )
    else:
        fig2.update_layout(title='Ranking de Cobertura Regional', template='plotly_white')
    return fig1, fig2


@app.callback(
    Output('download-regiao', 'data'),
    Input('btn-download', 'n_clicks'),
    State('region-select', 'value'),
    State('year-range', 'value'),
    prevent_initial_call=True
)
def download_regiao(n_clicks, selected_regions, year_range):
    if selected_regions is None or len(selected_regions) == 0:
        selected_regions = list(df_regiao['Regiao'].unique())
    yr0, yr1 = year_range
    df_f = df_regiao[(df_regiao['Regiao'].isin(selected_regions)) & (df_regiao['Ano'] >= yr0) & (df_regiao['Ano'] <= yr1)]
    return dcc.send_data_frame(df_f.to_csv, 'regiao_filtrada.csv', index=False)

if __name__ == '__main__':
    print("=" * 80)
    print("DASHBOARD SAÚDE MENTAL NO BRASIL")
    print("=" * 80)
    print("\nIniciando servidor Dash...")
    print("Acesse: http://127.0.0.1:8050")
    print("\nPara encerrar: Ctrl+C")
    print("=" * 80)
    
    # For deployment, prefer run_server; disable debug in production
    app.run(debug=True, host='127.0.0.1', port=8050)
