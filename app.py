import streamlit as st
from scipy.optimize import curve_fit
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
import datetime as dt
import pandas_datareader as pdr
import quandl
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import requests
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import expected_returns, risk_models, DiscreteAllocation, get_latest_prices
import os
import glob
from PIL import Image
from backtesting import Backtest, Strategy #pip install bactesting
from backtesting.lib import crossover #pip install bactesting
import pandas_ta as ta
import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import plotly.graph_objects as go
from hyperopt import fmin, tpe, hp, Trials
import hyperopt


# CLÉE API POUR LES FINNHUB
FINNHUB_API_KEY = 'cqo132hr01qo886587u0cqo132hr01qo886587ug'


@st.cache_data

# Fonction pour récupérer les actualités à partir du ticker
def get_stock_news(ticker):
    url = f'https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2023-01-01&to=2024-11-08&token={FINNHUB_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Impossible de récupérer les actualités pour {ticker}.")
        return None

# FONCTION POUR TRADUIRE (PAS FINI)
def translate_text(text, dest_language='en'):
    """Traduire le texte en utilisant Google Translate."""
    try:
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception as e:
        st.error(f"Erreur lors de la traduction : {e}")
        return text

# FONCTION POUR IMPORTER LES NOUVELLES FINNHUB
def get_finnhub_news():
    url = f'https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de la récupération des nouvelles.")
        return []

# FONCTION POUR AFFICHER LES NOUVELLES
def display_finnhub_news():
    news = get_finnhub_news()
    
    if news:
        st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>Actuatlités Financières</h1>", unsafe_allow_html=True)


        # Limiter à 5 articles les plus récents
        top_news = news[:5]

        for article in top_news:
            title = article.get('headline', 'Pas de titre')
            link = article.get('url', '#')
            summary = article.get('summary', 'Résumé non disponible')
            timestamp = article.get('datetime', '')
            formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'
            
            # Ajout d'une image d'illustration aléatoire pour chaque article (si disponible)
            image_url = article.get('image_url', 'https://via.placeholder.com/150')  # Placeholder si aucune image n'est disponible

            st.markdown(f"""
                <div style="margin-bottom: 20px; padding: 15px; border: 2px solid #e0e0e0; border-radius: 8px; background-color: #f7f7f7; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                    <div style="display: flex; align-items: center;">
                        <img src="{image_url}" style="width: 120px; height: 80px; object-fit: cover; border-radius: 8px; margin-right: 15px;" alt="Image d'illustration">
                        <div style="flex: 1;">
                            <h3 style="margin: 0; font-size: 18px; color: #1f77b4;">
                                <a href="{link}" target="_blank" style="text-decoration: none; color: inherit;">{title}</a>
                            </h3>
                            <p style="margin: 8px 0; color: #333;">{summary}</p>
                            <p style="margin: 0; font-size: 14px; color: #888;">{formatted_date}</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

# INITIALISATION DE LA SESSION DE SÉCURITÉ
if 'available_expirations' not in st.session_state:
    st.session_state.available_expirations = []

# FONCTION POUR TÉLÉCHARGER LES DONNÉES BOURSIÈRES
def download_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    info = stock.info
    stock_name = info.get('shortName', info.get('longName', ticker))
    return data, stock_name, info

# FONCTION POUR AFFICHER LES INFORMATIONS GÉNÉRALES DES ENTREPRISES
def display_company_info(info):
    st.write(f"**Nom**: {info.get('longName', 'N/A')}")
    st.write(f"**Symbole**: {info.get('symbol', 'N/A')}")
    st.write(f"**Nom Court**: {info.get('shortName', 'N/A')}")
    st.write(f"**Secteur**: {info.get('sector', 'N/A')}")
    st.write(f"**Industrie**: {info.get('industry', 'N/A')}")
    st.write(f"**Pays**: {info.get('country', 'N/A')}")
    st.write(f"**Description**: {info.get('longBusinessSummary', 'N/A')}")
    st.write(f"**Site Web**: {info.get('website', 'N/A')}")

def monte_carlo_simulation(data, num_simulations, num_days):
    # Calcul des rendements journaliers
    returns = (data['Close'] / data['Close'].shift(1) - 1).dropna()

    # Récupérer le dernier prix de clôture
    last_price = float(data['Close'].iloc[-1])  # Conversion en float pour éviter des erreurs
    
    # Initialiser un DataFrame pour stocker les résultats des simulations
    simulation_df = pd.DataFrame()

    # Effectuer les simulations de Monte Carlo
    for x in range(num_simulations):
        daily_vol = returns.std()  # Volatilité quotidienne
        price_series = [last_price]  # Initialisation de la série de prix avec le dernier prix

        for y in range(num_days):
            price = price_series[-1] * np.exp((0.5 * daily_vol**2) + daily_vol * norm.ppf(np.random.rand()))
            price_series.append(price)

        simulation_df[x] = price_series  # Ajouter les résultats de la simulation

    # Calculer la moyenne des simulations pour le dernier jour
    mean_price = simulation_df.iloc[-1].mean()

    return simulation_df, last_price, mean_price
# FONCTION POUR CALCULER LA VOLATILITÉ HISTORIQUE
def calculate_historical_volatility(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    historical_data = stock.history(start=start_date, end=end_date)
    returns = np.log(historical_data['Close'] / historical_data['Close'].shift(1))
    volatility = returns.std() * np.sqrt(252)
    return volatility

# FONCTION POUR LE FREE RISK RATE
def get_risk_free_rate():
    try:
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=365)
        risk_free_rate_data = pdr.get_data_fred('DGS10', start_date, end_date)
        latest_rate = risk_free_rate_data['DGS10'].iloc[-1] / 100
        return latest_rate
    except Exception as e:
        st.error(f"Erreur lors de la récupération du taux sans risque: {str(e)}")
        return None

# FONCTION POUR IMPORTER LES INFORMATIONS SUR LES OPTIONS
def fetch_option_data(ticker, expiry_date):
    stock = yf.Ticker(ticker)
    available_expirations = stock.options
    if expiry_date and expiry_date not in available_expirations:
        st.error(f"La date d'expiration `{expiry_date}` n'est pas disponible.")
        return None, None, None, None, None, available_expirations

    options = stock.option_chain(expiry_date) if expiry_date else None
    if options:
        calls = options.calls
        S = stock.history(period='1d')['Close'].iloc[-1]
        r = get_risk_free_rate() if get_risk_free_rate() is not None else 0.01
        strikes = calls['strike'].values
        market_prices = calls['lastPrice'].values
        expiration = dt.datetime.strptime(expiry_date, '%Y-%m-%d')
        T = (expiration - dt.datetime.now()).days / 365.0
        return S, strikes, market_prices, T, r, available_expirations
    return None, None, None, None, None, available_expirations

# FONCTION BACLK SHOLES POUR LES CALL
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# FONCTION POUR LES OPTIONS IMPLIED VOLATILITY 
def implied_volatility(S, K, T, r, market_price):
    def objective_function(sigma):
        return black_scholes_call(S, K, T, r, sigma) - market_price

    try:
        iv = brentq(objective_function, 1e-5, 3)
    except ValueError:
        iv = np.nan
    return iv

# FONCTION POUR LA PRÉDICTION DES VALEURS
def predict_stock_prices(ticker, forecast_days):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='2y')
    hist['Return'] = hist['Close'].pct_change()

    for lag in range(1, forecast_days + 1):
        hist[f'Lag{lag}'] = hist['Return'].shift(lag)
    hist.dropna(inplace=True)

    feature_cols = [f'Lag{lag}' for lag in range(1, forecast_days + 1)]
    X = hist[feature_cols]
    y = hist['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    forecast_features = np.array([hist[feature_cols].iloc[-1]])
    predicted_price = model.predict(forecast_features)

    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mean_price = y_test.mean()
    win_rate = 1 - (rmse / mean_price)

    return predicted_price, win_rate

# Fonction utilisant le Machine Learning pour prédire la valeur d'un actif
def predict_stock_prices_advanced(ticker, forecast_days):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1y')

    # Calculer le retour et les colonnes de lag
    hist['Return'] = hist['Close'].pct_change()
    hist = create_lagged_features(hist, forecast_days)

    # Séparer les caractéristiques et la cible
    feature_cols = [f'Lag{lag}' for lag in range(1, forecast_days + 1)]
    X = hist[feature_cols]
    y = hist['Close']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialiser et entraîner le modèle RandomForestRegressor
    model = RandomForestRegressor(n_estimators=1000, random_state=500)
    model.fit(X_train, y_train)

    # Prédire avec les dernières données disponibles
    latest_data = pd.DataFrame([X.iloc[-1]], columns=feature_cols)
    prediction = model.predict(latest_data)

    # Évaluer le modèle
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    win_rate = 1 - np.sqrt(mse) / y_test.mean()

    print(f"Predicted price for {ticker} in {forecast_days} days: ${prediction[0]:.2f}")
    print(f"Model Win Rate: {win_rate:.2%}")
    
    return prediction, win_rate

# Fonction KAN pour prédire le prix de l'action
def kan_model_price(t, P0, k, sigma0):
    # Modélisation hypothétique du prix
    return P0 * np.exp(k * t) * np.exp(sigma0 * np.sqrt(t))

# Function to plot prediction
def plot_prediction(ticker, forecast_days, predicted_price, win_rate):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1y')

    # Calculate 30-day moving average
    hist['30D_MA'] = hist['Close'].rolling(window=30).mean()

    # Define the future date
    future_date = hist.index[-1] + pd.Timedelta(days=forecast_days)

    fig = go.Figure()

    # Add candlestick chart for historical data
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name='Prix Historique'
    ))

    # Add 30-day moving average
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['30D_MA'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Moyenne Mobile 30 Jours'
    ))

    # Add predicted price as a scatter trace
    fig.add_trace(go.Scatter(
        x=[future_date],
        y=predicted_price,
        mode='markers',
        marker=dict(color='red', size=10),
        name=f'Prix Prédit dans {forecast_days} Jours'
    ))

    # Add annotation for predicted price
    fig.add_annotation(
        x=future_date,
        y=predicted_price[0],
        text=f'Prix Prédit: ${predicted_price[0]:.2f}\nTaux de Réussite: {win_rate:.2%}',
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(size=12, color='red'),
        align='center'
    )

    fig.update_layout(
        title=f'Prédiction des Prix de {ticker} pour les {forecast_days} Prochains Jours',
        xaxis_title='Date',
        yaxis_title='Prix',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='white',
        hovermode='x unified'
    )

    fig.add_annotation(
        text='StockGenius',
        xref='paper', yref='paper',
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(size=12, color='black'),
        align='left'
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

# Fonction pour ajouter une régression linéaire
def plot_linear_regression(data):
    # Prepare the data
    data = data.dropna().reset_index()
    data['Date_Ordinal'] = data['Date'].map(pd.Timestamp.toordinal)

    X = data[['Date_Ordinal']]
    y = data['Close']

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Create the Plotly figure
    fig = go.Figure()

    # Add historical price trace
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        mode='lines',
        name='Prix Historique',
        line=dict(color='blue')
    ))

    # Add regression line trace
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=predictions,
        mode='lines',
        name='Régression Linéaire',
        line=dict(color='red', dash='dash')
    ))

    # Update layout
    fig.update_layout(
        title='Régression Linéaire sur les Prix de Clôture' ,
        xaxis_title='Date',
        yaxis_title='Prix',
        plot_bgcolor='white',
        hovermode='x unified'
    )

    # Add annotation in the bottom left
    fig.add_annotation(
        text='StockGenius',
        xref='paper', yref='paper',
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(
            size=10,
            color='black'
        ),
        align='left',
        opacity=0.5
    )

    # Compute regression metrics
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)

    

    # Show the plot in Streamlit
    st.plotly_chart(fig)

def create_lagged_features(data, forecast_days):
    for i in range(1, forecast_days + 1):
        data[f'Lag{i}'] = data['Return'].shift(i)
    data = data.dropna()
    return data

# Fonction pour télécharger l'historique des prix de clôture ajustés
def get_price_history(ticker, sdate, edate):
    data = yf.download(ticker, start=sdate, end=edate)['Adj Close']
    return data

# Fonction pour afficher la performance des actions
def plot_performance(prices_df):
    fig = go.Figure()

    for c in prices_df.columns:
        fig.add_trace(go.Scatter(
            x=prices_df.index,
            y=prices_df[c],
            mode='lines',
            name=c
        ))

    fig.update_layout(
        title='Performance des Actions',
        xaxis_title='Date (Années)',
        yaxis_title='Prix USD (Clôture ajustée)',
        legend=dict(
            x=0,
            y=1,
            traceorder='normal',
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='rgba(0,0,0,0.5)',
            borderwidth=1
        ),
        plot_bgcolor='white',
        hovermode='x unified'
    )

    # Ajouter le texte dans le coin inférieur droit
    fig.add_annotation(
        text='StockGenius',
        xref='paper', yref='paper',
        x=0.99, y=0.01,
        showarrow=False,
        font=dict(
            size=10,
            color='black'
        ),
        align='right',
        opacity=0.5
    )

    # Afficher la grille pour l'axe des ordonnées
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Ajouter les outils d'interaction (zoom, dézoom, déplacement)
    fig.update_layout(
        dragmode='pan',  # Permet de déplacer le graphique
        xaxis=dict(
            rangeslider=dict(visible=True),  # Ajouter un curseur pour zoomer sur l'axe des x
            showspikes=True,  # Afficher les pointillés lors du survol
            spikemode='across',  # Les pointillés traversent l'axe
        ),
        yaxis=dict(
            showspikes=True,
            spikemode='across',
        )
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

# Fonction pour afficher la frontière efficiente
def plot_efficient_frontier(prices_df):
    returns_df = prices_df.pct_change()[1:]

    # Calcul du VaR historique
    confidence_level = 0.95
    VaR = returns_df.quantile(1 - confidence_level)

    # vecteur de rendement et matrice de covariance
    r = ((1 + returns_df).prod()) ** (252 / len(returns_df)) - 1
    cov = returns_df.cov() * 252
    e = np.ones(len(r))

    # calculer les rendements historiques moyens des actifs
    mu = expected_returns.mean_historical_return(prices_df)

    # Calculer la matrice de covariance échantillon des rendements des actifs
    S = risk_models.sample_cov(prices_df)
    S = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()
    S = (S + S.T) / 2

    # Créer un objet Frontière Efficiente
    ef = EfficientFrontier(mu, S)

    # optimiser pour le ratio de Sharpe maximum
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    latest_prices = get_latest_prices(prices_df)
    weights = cleaned_weights

    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=3000)
    allocation, leftover = da.greedy_portfolio()

    # Calculer la frontière efficiente
    icov = np.linalg.inv(cov)
    h = np.matmul(e, icov)
    g = np.matmul(r, icov)
    a = np.sum(e * h)
    b = np.sum(r * h)
    c = np.sum(r * g)
    d = a * c - b**2

    # portefeuille de tangence minimum et variance
    mvp = h / a
    mvp_returns = b / a
    mvp_risk = (1 / a) ** (1 / 2)

    # portefeuille de tangence
    tagency = g / b
    tagency_returns = c / b
    tagency_risk = c ** (1 / 2) / b

    min_expected_return = mu.min()
    max_expected_return = mu.max()
    exp_returns = np.linspace(min_expected_return, max_expected_return, num=100)
    risk = ((a * exp_returns ** 2 - 2 * b * exp_returns + c) / d) ** (1 / 2)

    # Tracé de la ligne de marché des titres (SML)
    SML_slope = 1 / c**(1 / 2)
    SML_risk = exp_returns * SML_slope

    # Création du graphique avec Plotly
    fig = go.Figure()

    # Frontière efficiente
    fig.add_trace(go.Scatter(x=risk, y=exp_returns, mode='lines', name='Frontière Efficiente',
                             line=dict(color='blue', dash='dash')))

    # Ligne de marché des titres (SML)
    fig.add_trace(go.Scatter(x=SML_risk, y=exp_returns, mode='lines', name='Ligne de Marché des Titres (SML)',
                             line=dict(color='red', dash='dashdot')))

    # Points des portefeuilles
    fig.add_trace(go.Scatter(x=[mvp_risk], y=[mvp_returns], mode='markers',
                             name='Portefeuille de Volatilité Minimale', marker=dict(color='red', size=10, symbol='star')))
    fig.add_trace(go.Scatter(x=[tagency_risk], y=[tagency_returns], mode='markers',
                             name='Portefeuille Optimal en Risque', marker=dict(color='green', size=10, symbol='star')))

    # Mise en forme du graphique
    fig.update_layout(
        title="Frontière Efficiente & Ligne de Marché des Titres",
        xaxis_title="Écart-type (Risque)",
        yaxis_title="Rendement Attendu",
        legend=dict(
            x=0.99, y=0.01,
            xanchor='right', yanchor='bottom',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.5)',
            font=dict(color='black')
        ),
        plot_bgcolor='white',
        hovermode='x unified'
    )

    # Ajouter une grille pour l'axe des ordonnées
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Ajouter le texte "guccipepito" en bas à gauche
    fig.add_annotation(
        text="StockGenius",
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(size=10, color="black"),
        opacity=0.5
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

    # Résolution du problème de rendement cible
    target_return = tagency_returns
    target_risk = tagency_risk

    if target_return < mvp_returns:
        optimal_portfolio = mvp
        optimal_return = mvp_returns
        optimal_risk = mvp_risk
    else:
        l = (c - b * target_return) / d
        m = (a * target_return - b) / d
        optimal_portfolio = l * h + m * g
        optimal_return = np.sum(optimal_portfolio * r)
        optimal_risk = ((a * optimal_return ** 2 - 2 * b * optimal_return + c) / d) ** (1 / 2)

    # Récupération des performances du portefeuille
    annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)

  

    # Texte des performances du portefeuille
    performance_text = f"Rendement annuel attendu : {annual_return * 100:.1f}%\n" \
                       f"Volatilité annuelle : {annual_volatility * 100:.1f}%\n" \
                       f"Ratio de Sharpe : {sharpe_ratio:.2f}"
    
    st.write(performance_text)
    
    
    # Création du graphique pour les performances du portefeuille et poids nettoyés
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].text(0.1, 0.5, performance_text, fontsize=12, ha='left', va='center')
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Performances du Portefeuille')

    # Graphique des poids nettoyés du portefeuille
    axs[0, 1].bar(weights.keys(), weights.values())
    axs[0, 1].set_title('Poids Nettoyés du Portefeuille')
    axs[0, 1].set_xlabel('Actifs')
    axs[0, 1].set_ylabel('Poids')
    axs[0, 1].tick_params(axis='x', rotation=45)

    # Graphique de l'allocation discrète des actifs
    axs[1, 0].pie(list(allocation.values()), labels=list(allocation.keys()), autopct='%1.1f%%', startangle=140)
    axs[1, 0].set_title('Allocation Discrète des Actifs')

    # Graphique des fonds restants après l'allocation
    axs[1, 1].text(0.5, 0.5, f"Fonds restants :\n{leftover:.2f} CAD", fontsize=14, ha='center', va='center')
    axs[1, 1].axis('off')
    axs[1, 1].set_title('Fonds Restants')

    # Ajout de la légende spécifique en bas à droite
    fig.text(0.95, 0.05, 'StockGenius', fontsize=12, color='black', ha='right', va='bottom', alpha=0.5)

    # Ajustement de l'espacement entre les sous-graphiques
    plt.tight_layout()

    # Affichage du graphique
    plt.show()
    

    # Impression de l'allocation discrète du portefeuille et des fonds restants
    st.write(f"Allocation Discrète du Portefeuille: {allocation}")
    st.write(f"Fonds Restants: {leftover:.2f}")

# Function to plot volatility surface
def plot_volatility_surface(ticker, expiry_date, forecast_days):
    S, strikes, market_prices, T, r, _ = fetch_option_data(ticker, expiry_date)
    if S is None:
        return

    ivs_list = []
    maturities_list = []

    for day in range(forecast_days + 1):
        T_forecast = T + day / 365.0
        ivs = [implied_volatility(S, K, T_forecast, r, P) for K, P in zip(strikes, market_prices)]
        ivs_list.append(ivs)
        maturities_list.append([T_forecast] * len(strikes))

    ivs_array = np.array(ivs_list)
    maturities_array = np.array(maturities_list)

    if ivs_array.size == 0 or strikes.size == 0:
        st.write("Données insuffisantes pour tracer la surface.")
        return

    strike_grid, maturity_grid = np.meshgrid(strikes, np.linspace(T, T + forecast_days / 365.0, forecast_days + 1))
    fig = go.Figure(data=[go.Surface(z=ivs_array, x=strike_grid, y=maturity_grid, colorscale='Viridis')])
    fig.update_layout(
        title=f'Surface de Volatilité Implicite pour {ticker}',
        scene=dict(
            xaxis_title='Prix d\'Exercice',
            yaxis_title='Échéance (Années)',
            zaxis_title='Volatilité Implicite'
        )
    )
    fig.add_annotation(
        text="StockGenius",
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(size=10, color="white"),
        opacity=0.5
    )
    st.plotly_chart(fig)

def download_bond_data(ticker, start_date, end_date):
    bond = yf.Ticker(ticker)
    data = bond.history(start=start_date, end=end_date)
    bond_name = bond.info.get('shortName', bond.info.get('longName', ticker))
    return data['Close'], bond_name, bond.info

def plot_sinusoidal_with_ticker(ticker, start_date, end_date, amplitude=1.0, period=30):
    """
    Trace les prix de l'action avec une vague sinusoïdale superposée.
    
    Paramètres :
    - ticker : Le symbole de l'action (par ex. 'AAPL').
    - start_date : Date de début des données historiques.
    - end_date : Date de fin des données historiques.
    - amplitude : Amplitude de la vague sinusoïdale.
    - period : Période de la vague sinusoïdale.
    """
    # Télécharger les données historiques du ticker
    data = yf.download(ticker, start=start_date, end=end_date)
    prices = data['Close']
    
    # Calculer les temps en jours
    time = np.arange(len(prices))
    
    # Générer la vague sinusoïdale
    sinusoidal_wave = amplitude * np.sin(2 * np.pi * time / period)
    
    # Créer un graphique Plotly
    fig = go.Figure()
    
    # Tracer les prix de l'action
    fig.add_trace(go.Scatter(
        x=prices.index, 
        y=prices,
        mode='lines',
        name='Prix de l\'action',
        line=dict(color='blue')
    ))
    
    # Tracer la vague sinusoïdale
    fig.add_trace(go.Scatter(
        x=prices.index, 
        y=sinusoidal_wave + np.mean(prices),
        mode='lines',
        name='Vague sinusoïdale',
        line=dict(color='red', dash='dash')
    ))
    
    # Mettre à jour la mise en page du graphique
    fig.update_layout(
        title=f'Vague Sinusoïdale et Prix de l\'Action: {ticker}',
        xaxis_title='Temps',
        yaxis_title='Prix',
        plot_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True),
        width=800,
        height=500
    )
    
    # Afficher le graphique
    fig.show()

def get_stock_prices(tickers):
    prices = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')  # Dernier jour de données
        if not data.empty:
            prices[ticker] = f"${data['Close'].iloc[-1]:,.2f}"  # Formater le prix
        else:
            prices[ticker] = "N/A"
    return prices

def filter_option_trading_news(news_list):
    """Filtre les nouvelles pour ne garder que celles qui parlent de trading d'options."""
    filtered_news = [article for article in news_list if 'option' in article.get('headline', '').lower()]
    return filtered_news

def display_finnhub_news():
    news = get_finnhub_news()
    if news:
        st.title(f':newspaper: Actualités Financières')
        # Limiter à 10 articles les plus récents
        top_news = news[:5]
        for article in top_news:
            title = article.get('headline', 'Pas de titre')
            link = article.get('url', '#')
            summary = article.get('summary', 'Résumé non disponible')
            timestamp = article.get('datetime', '')
            formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'

            st.markdown(f"""
                <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px;">
                    <h3 style="margin: 0; font-size: 16px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                    <p style="margin: 5px 0; color: #555;">{summary}</p>
                    <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

def display_options_news():
    news = get_finnhub_news()
    st.subheader('Top 5 Actualités sur le Trading d\'Options')

    if news:
        option_news = filter_option_trading_news(news)
        if option_news:
            # Limiter à 5 articles
            top_news = option_news[:5]
            for article in top_news:
                title = article.get('headline', 'Pas de titre')
                link = article.get('url', '#')
                summary = article.get('summary', 'Résumé non disponible')
                timestamp = article.get('datetime', '')
                formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'
                image_url = article.get('image', '')

                st.markdown(f"""
                    <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9;">
                        <div style="display: flex; align-items: center;">
                            <img src="{image_url}" alt="Image" style="width: 100px; height: auto; margin-right: 10px; border-radius: 5px;">
                            <div>
                                <h3 style="margin: 0; font-size: 18px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                                <p style="margin: 5px 0; color: #555;">{summary}</p>
                                <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucune nouvelle sur le trading d'options disponible pour le moment.")
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

def display_economic_news():
    news = get_finnhub_news()
    st.subheader('Top 5 Actualités Économiques')

    if news:
        # Filtrer les nouvelles économiques
        economic_news = [article for article in news if 'economy' in article.get('headline', '').lower()]
        # Limiter à 5 articles
        top_news = economic_news[:5]
        
        if top_news:
            for article in top_news:
                title = article.get('headline', 'Pas de titre')
                link = article.get('url', '#')
                summary = article.get('summary', 'Résumé non disponible')
                timestamp = article.get('datetime', '')
                formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'
                image_url = article.get('image', '')

                st.markdown(f"""
                    <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9;">
                        <div style="display: flex; align-items: center;">
                            <img src="{image_url}" alt="Image" style="width: 100px; height: auto; margin-right: 10px; border-radius: 5px;">
                            <div>
                                <h3 style="margin: 0; font-size: 18px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                                <p style="margin: 5px 0; color: #555;">{summary}</p>
                                <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucune nouvelle économique disponible pour le moment.")
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

def filter_bond_market_news(news_list):
    """Filtre les nouvelles pour ne garder que celles sur le marché obligataire."""
    filtered_news = [article for article in news_list if 'bond' in article.get('headline', '').lower() or 'obligation' in article.get('headline', '').lower()]
    return filtered_news

def display_bond_market_news():
    news = get_finnhub_news()
    st.subheader('Top 5 Actualités sur le Marché Obligataire')

    if news:
        bond_news = filter_bond_market_news(news)
        # Limiter à 5 articles
        top_news = bond_news[:5]
        
        if top_news:
            for article in top_news:
                title = article.get('headline', 'Pas de titre')
                link = article.get('url', '#')
                summary = article.get('summary', 'Résumé non disponible')
                timestamp = article.get('datetime', '')
                formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'
                image_url = article.get('image', '')

                st.markdown(f"""
                    <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9;">
                        <div style="display: flex; align-items: center;">
                            <img src="{image_url}" alt="Image" style="width: 100px; height: auto; margin-right: 10px; border-radius: 5px;">
                            <div>
                                <h3 style="margin: 0; font-size: 18px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                                <p style="margin: 5px 0; color: #555;">{summary}</p>
                                <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucune nouvelle sur le marché obligataire disponible pour le moment.")
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

def display_finnhub_news_ticker(ticker):
    news = get_finnhub_news_ticker
    if news:
        st.subheader(f'Actualités pour {ticker}')
        # Limiter à 5 articles les plus récents
        top_news = news[:5]
        for article in top_news:
            title = article.get('headline', 'Pas de titre')
            link = article.get('url', '#')
            summary = article.get('summary', 'Résumé non disponible')
            timestamp = article.get('datetime', '')
            formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'

            st.markdown(f"""
                <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px;">
                    <h3 style="margin: 0; font-size: 16px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                    <p style="margin: 5px 0; color: #555;">{summary}</p>
                    <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

def get_finnhub_news_ticker(ticker):
    url = f'https://finnhub.io/api/v1/company-news?symbol={ticker}&from={dt.date.today() - dt.timedelta(days=30)}&to={dt.date.today()}&token={FINNHUB_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de la récupération des nouvelles.")
        return []

def display_excel_file(file_path=None):
    # Option pour uploader un fichier si file_path n'est pas fourni
    if file_path is None:
        uploaded_file = st.file_uploader("Choisissez un fichier Excel", type=["xlsx"])
        if uploaded_file is not None:
            file_path = uploaded_file
        else:
            st.warning("Veuillez télécharger un fichier Excel.")
            return  # Sortie anticipée si aucun fichier n'est fourni

    try:
        # Lire le fichier Excel
        df = pd.read_excel(file_path)

        # Appliquer un style professionnel avec pandas
        styled_df = df.style.set_properties(**{
            'background-color': '#f5f5f5',
            'color': '#333333',
            'border-color': 'black',
            'font-size': '12pt',
            'text-align': 'center',
        }).format(precision=2)  # Ajuster le formatage, par exemple, 2 décimales

        # Ajouter un titre et une description
        st.title(f":ballot_box_with_ballot: Liste d'entreprises à surveiller")
        #st.write("Ce tableau présente les données extraites du fichier Excel, formatées pour une meilleure lisibilité.")

        # Afficher le tableau avec Streamlit
        st.dataframe(styled_df, use_container_width=True)

    except FileNotFoundError:
        st.error(f"Le fichier '{file_path}' est introuvable. Veuillez vérifier le chemin.")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la lecture du fichier : {str(e)}")


    

def get_price_change(ticker, period="1mo"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    # Vérifier s'il y a des données historiques disponibles
    if len(hist) == 0:
        return None
    
    # Calculer le changement de prix entre le premier et le dernier jour
    first_price = hist['Close'].iloc[0]
    last_price = hist['Close'].iloc[-1]
    change = (last_price - first_price) / first_price
    
    return change

def get_market_trend(change):
    if change > 0.05:
        return "BULL"
    elif change < -0.05:
        return "BEAR"
    else:
        return "HOLD"
    
def create_gauge(change):
    trend = get_market_trend(change)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = change,
        gauge = {
            'axis': {'range': [-1, 1]},
            'steps' : [
                {'range': [-1, -0.05], 'color': "red"},
                {'range': [-0.05, 0.05], 'color': "gray"},
                {'range': [0.05, 1], 'color': "green"}],
            'threshold' : {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': change}
        },
        title = {'text': trend},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    
    return fig

def display_financial_summary(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    st.subheader("Résumé des Informations Financières")
    st.write(f"**Beta de l'action:** {info.get('beta', 'N/A')}")
    st.write(f"**Ratio cours/bénéfices (P/E) sur les bénéfices attendus:** {info.get('forwardPE', 'N/A')}")
    st.write(f"**Bénéfice par action (EPS) sur les bénéfices attendus:** {info.get('forwardEps', 'N/A')}")
    st.write(f"**Marges bénéficiaires:** {info.get('profitMargins', 'N/A')}")
    st.write(f"**Retour sur les actifs (ROA):** {info.get('returnOnAssets', 'N/A')}")
    st.write(f"**Valeur comptable):** {info.get('bookValue', 'N/A')}")
    st.write(f"**Ratio prix/valeur comptable):** {info.get('priceToBook', 'N/A')}")
    st.write(f"**Nombre d'employés à plein temps):** {info.get('fullTimeEmployees', 'N/A')}")

def download_futures_data(ticker, start_date, end_date, progress=False):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['MA30'] = data['Close'].rolling(window=30).mean()  # Calcul de la moyenne mobile sur 3 jours
    return data

def plot_futures_data(data, ticker):
    fig = go.Figure()

    # Ajouter le graphique en bougies
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name=f'{ticker} Candlesticks'))

    # Ajouter la moyenne mobile sur 3 jours
    fig.add_trace(go.Scatter(x=data.index, 
                             y=data['MA30'], 
                             mode='lines', 
                             line=dict(color='blue', width=2),
                             name='MA30 (30 jours)'))

    # Mise à jour du layout pour ajouter les titres et les légendes
    fig.update_layout(
        title=f'{ticker} - Prix Historique avec Moyenne Mobile',
        xaxis_title='Date',
        yaxis_title='Prix USD',
        legend=dict(
            x=0,
            y=1,
            traceorder='normal',
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='rgba(0,0,0,0.5)',
            borderwidth=1
        ),
        plot_bgcolor='white',
        hovermode='x unified'
    )

    # Ajouter le texte 'guccipepito' dans le coin inférieur gauche
    fig.add_annotation(
        text='StockGenius',
        xref='paper', yref='paper',
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(
            size=10,
            color='black'
        ),
        align='left',
        opacity=0.5
    )

    st.plotly_chart(fig)

def display_futures_news():
    news = get_finnhub_news()  # Assurez-vous que cette fonction est définie pour obtenir les nouvelles
    st.subheader('Top 5 Actualités sur les Futures')

    if news:
        futures_news = get_finnhub_news()  # Assurez-vous que cette fonction filtre les nouvelles sur les futures
        if futures_news:
            # Limiter à 5 articles
            top_news = futures_news[:5]
            for article in top_news:
                title = article.get('headline', 'Pas de titre')
                link = article.get('url', '#')
                summary = article.get('summary', 'Résumé non disponible')
                timestamp = article.get('datetime', '')
                formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'
                image_url = article.get('image', '')

                st.markdown(f"""
                    <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9;">
                        <div style="display: flex; align-items: center;">
                            <img src="{image_url}" alt="Image" style="width: 100px; height: auto; margin-right: 10px; border-radius: 5px;">
                            <div>
                                <h3 style="margin: 0; font-size: 18px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                                <p style="margin: 5px 0; color: #555;">{summary}</p>
                                <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucune nouvelle sur les futures disponible pour le moment.")
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

def get_forex_data(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        latest_data = data.iloc[-1]
        return {
            'Ticker': ticker,
            'Close': latest_data['Close'],
            'Open': latest_data['Open'],
            'High': latest_data['High'],
            'Low': latest_data['Low'],
            'Volume': latest_data['Volume'],
            'Date': latest_data.name
        }
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {e}")
        return None

def display_forex_news():
    news = get_finnhub_news()  # Obtenez les nouvelles sur le forex
    st.subheader('Top 5 Actualités sur le Forex')

    if news:
        # Limiter à 5 articles
        top_news = news[:5]
        for article in top_news:
            title = article.get('headline', 'Pas de titre')
            link = article.get('url', '#')
            summary = article.get('summary', 'Résumé non disponible')
            timestamp = article.get('datetime', '')
            formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'
            image_url = article.get('image', '')

            st.markdown(f"""
                <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9;">
                    <div style="display: flex; align-items: center;">
                        <img src="{image_url}" alt="Image" style="width: 100px; height: auto; margin-right: 10px; border-radius: 5px;">
                        <div>
                            <h3 style="margin: 0; font-size: 18px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                            <p style="margin: 5px 0; color: #555;">{summary}</p>
                            <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

def display_option_data(strikes, market_prices, ivs):
    # Titre stylisé

    # Création du DataFrame
    option_data = pd.DataFrame({
        'Strike': strikes,
        'Prix du marché': market_prices,
        'Volatilité implicite': ivs
    })

    # Application de styles au DataFrame
    styled_option_data = option_data.style.set_properties(**{
        'background-color': '#f0f0f0',
        'color': '#333',
        'border-color': 'black',
        'font-size': '12pt',
        'text-align': 'center',
    }).format({
        'Strike': '{:.2f}',
        'Prix du marché': '{:.2f} $',
        'Volatilité implicite': '{:.2%}'
    })

    # Affichage du tableau stylisé
    st.dataframe(styled_option_data, use_container_width=True)

    # Ajout de l'annotation "StockGenius" en bas à gauche
    st.markdown(
        "<div style='text-align: left; color: #888; font-size: 10pt; margin-top: 10px;'>StockGenius</div>",
        unsafe_allow_html=True
    )

# Fonction pour sauvegarder les données dans un fichier CSV sans écraser les données existantes
def save_to_csv(data_dict, folder='KAN_pred', filename='predictions.csv'):
    # Créer le dossier s'il n'existe pas
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filepath = os.path.join(folder, filename)
    
    # Convertir les données en DataFrame
    df = pd.DataFrame([data_dict])
    
    # Vérifier si le fichier existe
    file_exists = os.path.isfile(filepath)
    
    # Si le fichier existe, ajouter les données sans écraser
    if file_exists:
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        # Si le fichier n'existe pas, écrire le fichier avec un header
        df.to_csv(filepath, mode='w', header=True, index=False)

# Fonction pour enregistrer les résultats du backtest dans un fichier CSV
def save_backtest_results_to_csv(ticker, results, trades,folder='test1', filename='backtest_results.csv'):
    # Convertir les résultats en DataFrame
    result_data = {
        'Ticker': ticker,
        'Equity Final': [results['Equity Final [$]']],
        'Sharpe Ratio': [results['Sharpe Ratio']],
        'Start Date': [results['Start']],
        'End Date': [results['End']],
        'Total Trades': [results['# Trades']]
    }
    
    trades_data = trades.copy()
    trades_data['Ticker'] = ticker

    result_df = pd.DataFrame(result_data)

    # Vérifier si le fichier existe déjà
    if not os.path.exists(filename):
        # Si le fichier n'existe pas, créer un nouveau fichier avec l'en-tête
        result_df.to_csv(filename, mode='w', index=False)
        trades_data.to_csv(f"trades_{ticker}.csv", mode='w', index=False)
    else:
        # Si le fichier existe, ajouter les données sans écraser
        result_df.to_csv(filename, mode='a', header=False, index=False)
        trades_data.to_csv(f"trades_{ticker}.csv", mode='a', header=False, index=False)

# Page d'authentification
def login():
    
    st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>StockGenius ©</h1>", unsafe_allow_html=True)
    st.title(f":mailbox: Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")

    
    


    if st.button("Se connecter"):
        if username == "admin" and password == "password":  # Remplacez par votre système d'authentification
            st.success("Connexion réussie!")
            st.session_state.authenticated = True
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect")
     
# Vérifiez si l'utilisateur est authentifié
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login()
else:
    # Streamlit app
    st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>StockGenius ©</h1>", unsafe_allow_html=True)
    

    # Sidebar
    #st.sidebar.title('Menu')
    app_mode = st.sidebar.selectbox(':mag_right: Choisissez une section',
                                    ['Accueil', 'Recherche', 'Options', 'Prédictions', 'Gestion des Actifs', 'Backtesting - StockGenius', 'Contact'
                                        ])

    # Tabs content
    if app_mode == 'Accueil':
        
        # Sidebar
        st.sidebar.title(f':speaking_head_in_silhouette: Outils')
        st.sidebar.markdown("""
            <ul style="list-style-type:none;">
                <li><a href="https://tubitv.com/live/400000081/bloomberg-tv" target="_blank" style="color: #FFFFFF; text-decoration: none;">Bloomberg TV</a></li>
                <li><a href="https://www.tradingview.com" target="_blank" style="color: #FFFFFF; text-decoration: none;">TradingView</a></li>
                <li><a href="https://www.questrade.com" target="_blank" style="color: #FFFFFF; text-decoration: none;">Questrade</a></li>
                <li><a href="https://valueinvesting.io" target="_blank" style="color: #FFFFFF; text-decoration: none;">Value Investing</a></li>
                <li><a href="https://docs.google.com/spreadsheets/d/1JhJZqu25MMCHz1of7sAIX3FtBpzCfQdluNQzsciy14k/edit?gid=1161698254#gid=1161698254" target="_blank" style="color: #FFFFFF; text-decoration: none;">États financiers</a></li>
                <li><a href="https://www.forexfactory.com/calendar" target="_blank" style="color: #FFFFFF; text-decoration: none;">Forex Factory</a></li>
                <li><a href="https://finviz.com/map.ashx?t=sec" target="_blank" style="color: #FFFFFF; text-decoration: none;">Carte des Marchés</a></li>
                <li><a href="https://colab.research.google.com/drive/1ORJR3YMtvzMOBvZ1HzUqVpe1LwTb6FRJ" target="_blank" style="color: #FFFFFF; text-decoration: none;">Google Colab</a></li>

            </ul>
            """, unsafe_allow_html=True)
        
    
        # Code HTML et CSS pour le défilement infini des logos
        scrolling_logos = """
            <style>
            @keyframes scroll {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
            }

            .scrolling-container {
            width: 100%;
            overflow: hidden;
            position: relative;
            background-color: #f8f9fa; /* Couleur de fond, optionnelle */
            padding: 10px 0;
            }

            .scrolling-logos {
            display: flex;
            justify-content: center;
            align-items: center;
            animation: scroll 40s linear infinite; /* 40s pour une vitesse plus lente */
            }

            .scrolling-logos img {
            height: 50px; /* Taille des logos */
            margin: 0 20px; /* Espacement entre les logos */
            }
            </style>

            <div class="scrolling-container">
            <div class="scrolling-logos">
                <img src="https://upload.wikimedia.org/wikipedia/commons/1/1a/Seal_of_the_United_States_Federal_Reserve_System.svg" alt="FOMC">
                <img src="https://upload.wikimedia.org/wikipedia/commons/8/87/NASDAQ_Logo.svg" alt="NASDAQ">
                <img src="https://upload.wikimedia.org/wikipedia/commons/b/be/NYSE_Logo.svg" alt="NYSE">
                <img src="https://upload.wikimedia.org/wikipedia/commons/b/bd/TSX_Logo.svg" alt="TSX">
                <img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Bloomberg_logo.svg" alt="Bloomberg">
                <img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Yahoo%21_%282019%29.svg" alt="Yahoo">
                <img src="https://upload.wikimedia.org/wikipedia/en/0/0c/MarketWatch_Logo.svg" alt="MarketWatch">
                <img src="https://store-images.s-microsoft.com/image/apps.18771.7c58e601-ae93-41a7-8654-f7243c835eee.097f998a-684c-49c1-b8de-e12877c70b81.a4ef990c-549c-488b-a2c7-aa6b455b672f.png" alt="ValueInvesting.io">
                <img src="https://1000logos.net/wp-content/uploads/2023/11/Forex-Factory-Logo.jpg" alt="Forex Factory">
                <img src="https://upload.wikimedia.org/wikipedia/commons/e/e3/CNBC_logo.svg" alt="CNBC">
                <img src="https://upload.wikimedia.org/wikipedia/commons/3/37/Investopedia_Logo.svg" alt="INVESTOPEDIA">
                <img src="https://upload.wikimedia.org/wikipedia/fr/d/d1/Les_affaires_%28logo%29.png" alt="LESAFFAIRES">
                <img src="https://www.finance-investissement.com/wp-content/uploads/sites/2/2018/01/fi-logo.svg" alt="Finance">
            </div>
            </div>
            """

        


        import streamlit as st
        import pandas as pd
        import plotly.express as px
        from urllib.parse import quote

        # Fonction pour charger les données depuis Wikipedia avec st.cache_data
        def load_data():
            url = "https://fr.wikipedia.org/wiki/Liste_de_pays_par_dette_extérieure"
            
            # Encoder les caractères non-ASCII dans l'URL
            encoded_url = quote(url, safe=':/')
            
            # Lire le tableau depuis Wikipedia
            tables = pd.read_html(encoded_url)
            
            # Sélectionner le tableau pertinent
            debt_data = tables[0]
            
            # Renommer les colonnes pour correspondre au format attendu
            debt_data.columns = ['Rank', 'Country', 'Total Debt (US$)', 'Per capita (US$)', 'Date', 'Source']

            # Remplacer les espaces insécables (\xa0) et autres caractères non numériques dans 'Total Debt (US$)'
            debt_data['Total Debt (US$)'] = debt_data['Total Debt (US$)'].replace({'\$': '', ',': '', '\xa0': ''}, regex=True)

            # Convertir les valeurs en float
            debt_data['Total Debt (US$)'] = pd.to_numeric(debt_data['Total Debt (US$)'], errors='coerce')

            # Mapper les noms de pays pour correspondre au format attendu par Plotly
            country_mapping = {
                "États-Unis": "United States",
                "Chine": "China",
                "Union européenne": "European Union",
                "Royaume-Uni": "United Kingdom",
                "Allemagne": "Germany",
                "France": "France",
                "Japon": "Japan",
                "Irlande": "Ireland",
                "Norvège": "Norway",
                "Italie": "Italy",
                "Espagne": "Spain",
                "Luxembourg": "Luxembourg",
                "Belgique": "Belgium",
                "Suisse": "Switzerland",
                "Australie": "Australia",
                "Canada": "Canada",
                "Suède": "Sweden",
                "Autriche": "Austria",
                "Hong Kong": "Hong Kong",
                "Danemark": "Denmark",
                "Grèce": "Greece",
                "Portugal": "Portugal",
                "Russie": "Russia",
                "Pays-Bas": "Netherlands",
                "Finlande": "Finland",
                "Corée du Sud": "South Korea",
                "Brésil": "Brazil",
                "Turquie": "Turkey",
                "Pologne": "Poland",
                "Inde": "India",
                "Mexique": "Mexico",
                "Indonésie": "Indonesia",
                "Argentine": "Argentina",
                "Hongrie": "Hungary",
                "Émirats arabes unis": "United Arab Emirates",
                "Roumanie": "Romania",
                "Ukraine": "Ukraine",
                "Kazakhstan": "Kazakhstan",
                "Israël": "Israel",
                "République tchèque": "Czech Republic",
                "Chili": "Chile",
                "Arabie saoudite": "Saudi Arabia",
                "Thaïlande": "Thailand",
                "Afrique du Sud": "South Africa",
                "Maroc": "Morocco",
                "Malaisie": "Malaysia",
                "Qatar": "Qatar",
                "Nouvelle-Zélande": "New Zealand",
                "Philippines": "Philippines",
                "Croatie": "Croatia",
                "Slovaquie": "Slovakia",
                "Colombie": "Colombia",
                "Pakistan": "Pakistan",
                "Koweït": "Kuwait",
                "Venezuela": "Venezuela",
                "Irak": "Iraq",
                "Slovénie": "Slovenia",
                "Bulgarie": "Bulgaria",
                "Soudan": "Sudan",
                "Lettonie": "Latvia",
                "Liban": "Lebanon",
                "Viêt Nam": "Vietnam",
                "Pérou": "Peru",
                "Chypre": "Cyprus",
                "Serbie": "Serbia",
                "Égypte": "Egypt",
                "Lituanie": "Lithuania",
                "Biélorussie": "Belarus",
                "Bangladesh": "Bangladesh",
                "Estonie": "Estonia",
                "Singapour": "Singapore",
                "Cuba": "Cuba",
                "Tunisie": "Tunisia",
                "Monaco": "Monaco",
                "Angola": "Angola",
                "Sri Lanka": "Sri Lanka",
                "Guatemala": "Guatemala",
                "Équateur": "Ecuador",
                "Bahreïn": "Bahrain",
                "Panama": "Panama",
                "République démocratique du Congo": "Democratic Republic of the Congo",
                "Uruguay": "Uruguay",
                "République dominicaine": "Dominican Republic",
                "Iran": "Iran",
                "Jamaïque": "Jamaica",
                "Corée du Nord": "North Korea",
                "Côte d'Ivoire": "Ivory Coast",
                "Salvador": "El Salvador",
                "Nigeria": "Nigeria",
                "Oman": "Oman",
                "Costa Rica": "Costa Rica",
                "Bosnie-Herzégovine": "Bosnia and Herzegovina",
                "Kenya": "Kenya",
                "Syrie": "Syria",
                "Zimbabwe": "Zimbabwe",
                "Tanzanie": "Tanzania",
                "Yémen": "Yemen",
                "Birmanie": "Myanmar",
                "Ghana": "Ghana",
                "Libye": "Libya",
                "Malte": "Malta",
                "Laos": "Laos",
                "Jordanie": "Jordan",
                "Macédoine": "North Macedonia",
                "Arménie": "Armenia",
                "Maurice": "Mauritius",
                "Turkménistan": "Turkmenistan",
                "République du Congo": "Republic of the Congo",
                "Mozambique": "Mozambique",
                "Moldavie": "Moldova",
                "Népal": "Nepal",
                "Cambodge": "Cambodia",
                "Trinité-et-Tobago": "Trinidad and Tobago",
                "Éthiopie": "Ethiopia",
                "Ouzbékistan": "Uzbekistan",
                "Nicaragua": "Nicaragua",
                "Sénégal": "Senegal",
                "Kirghizistan": "Kyrgyzstan",
                "Honduras": "Honduras",
                "Zambie": "Zambia",
                "Géorgie": "Georgia",
                "Cameroun": "Cameroon",
                "Azerbaïdjan": "Azerbaijan",
                "Liberia": "Liberia",
                "Islande": "Iceland",
                "Guinée": "Guinea",
                "Somalie": "Somalia",
                "Madagascar": "Madagascar",
                "Bénin": "Benin",
                "Ouganda": "Uganda",
                "Bolivie": "Bolivia",
                "Albanie": "Albania",
                "Mali": "Mali",
                "Afghanistan": "Afghanistan",
                "Paraguay": "Paraguay",
                "Gabon": "Gabon",
                "Namibie": "Namibia",
                "Botswana": "Botswana",
                "Niger": "Niger",
                "Burkina Faso": "Burkina Faso",
                "Tadjikistan": "Tajikistan",
                "Mongolie": "Mongolia",
                "Tchad": "Chad",
                "Sierra Leone": "Sierra Leone",
                "Papouasie-Nouvelle-Guinée": "Papua New Guinea",
                "Seychelles": "Seychelles",
                "Malawi": "Malawi",
                "Burundi": "Burundi",
                "République centrafricaine": "Central African Republic",
                "Palestine": "Palestine",
                "Belize": "Belize",
                "Érythrée": "Eritrea",
                "Maldives": "Maldives",
                "Guinée-Bissau": "Guinea-Bissau",
                "Bhoutan": "Bhutan",
                "Guinée équatoriale": "Equatorial Guinea",
                "Guyana": "Guyana",
                "Barbade": "Barbados",
                "Monténégro": "Montenegro",
                "Lesotho": "Lesotho",
                "Gambie": "Gambia",
                "Suriname": "Suriname",
                "Eswatini / Swaziland": "Eswatini",
                "Saint-Vincent-et-les-Grenadines": "Saint Vincent and the Grenadines",
                "Aruba": "Aruba",
                "Djibouti": "Djibouti",
                "Antigua-et-Barbuda": "Antigua and Barbuda",
                "Haïti": "Haiti",
                "Grenade": "Grenada",
                "Bahamas": "Bahamas",
                "Cap-Vert": "Cape Verde",
                "Saint-Kitts-et-Nevis": "Saint Kitts and Nevis",
                "Saint-Marin": "San Marino",
                "Îles Marshall": "Marshall Islands",
                "Vanuatu": "Vanuatu",
                "Sao Tomé-et-Principe": "Sao Tome and Principe",
                "Comores": "Comoros",
                "Kiribati": "Kiribati",
                "Samoa": "Samoa",
                "Tuvalu": "Tuvalu",
                "Mali": "Mali",
                "Bermudes": "Bermuda",
                "Seychelles": "Seychelles",
                "Grenade": "Grenada",
                "Antigua-et-Barbuda": "Antigua and Barbuda",
                "Géorgie": "Georgia"
            }

            # Appliquer le mapping des noms de pays
            debt_data['Country'] = debt_data['Country'].map(country_mapping).fillna(debt_data['Country'])

            return debt_data

        # Charger les données
        debt_data = load_data()

        # Graphique interactif avec Plotly
        fig = px.choropleth(
            debt_data,
            locations='Country',
            locationmode='country names',
            color='Total Debt (US$)',
            hover_name='Country',
            title='Dette publique par pays - COVID-19',
            color_continuous_scale=px.colors.sequential.Plasma
        )

        # Add annotation in the bottom left
        fig.add_annotation(
            text='StockGenius',
            xref='paper', yref='paper',
            x=0.01, y=0.01,
            showarrow=False,
            font=dict(
                size=10,
                color='black'
            ),
            align='left',
            opacity=1
        )

    

        import streamlit as st
        import pandas as pd
        import math
        from pathlib import Path

        

        # -----------------------------------------------------------------------------
        # Declare some useful functions.

        @st.cache_data
        def get_gdp_data():
            """Grab GDP data from a CSV file.

            This uses caching to avoid having to read the file every time. If we were
            reading from an HTTP endpoint instead of a file, it's a good idea to set
            a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
            """

            # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
            DATA_FILENAME = Path(__file__).parent/'gdp_data.csv'
            raw_gdp_df = pd.read_csv(DATA_FILENAME)

            MIN_YEAR = 1960
            MAX_YEAR = 2022

            # The data above has columns like:
            # - Country Name
            # - Country Code
            # - [Stuff I don't care about]
            # - GDP for 1960
            # - GDP for 1961
            # - GDP for 1962
            # - ...
            # - GDP for 2022
            #
            # ...but I want this instead:
            # - Country Name
            # - Country Code
            # - Year
            # - GDP
            #
            # So let's pivot all those year-columns into two: Year and GDP
            gdp_df = raw_gdp_df.melt(
                ['Country Code'],
                [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
                'Year',
                'GDP',
            )

            # Convert years from string to integers
            gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

            return gdp_df

        gdp_df = get_gdp_data()

        # -----------------------------------------------------------------------------
        # Draw the actual page

        # Set the title that appears at the top of the page.
        '''
        # :classical_building: Produit Intérieur Brut (PIB) par Pays

        '''

        # Add some spacing
        ''
        ''

        min_value = gdp_df['Year'].min()
        max_value = gdp_df['Year'].max()

        from_year, to_year = st.slider(
            'Temps',
            min_value=min_value,
            max_value=max_value,
            value=[min_value, max_value])

        countries = gdp_df['Country Code'].unique()

        if not len(countries):
            st.warning("Select at least one country")

        selected_countries = st.multiselect(
            'Pays',
            countries,
            ['CAN', 'USA', 'CHN', 'FRA', 'MEX', 'JPN'])

        ''
        ''
        ''

        # Filter the data
        filtered_gdp_df = gdp_df[
            (gdp_df['Country Code'].isin(selected_countries))
            & (gdp_df['Year'] <= to_year)
            & (from_year <= gdp_df['Year'])
        ]

        st.header('PIB depuis 2022', divider='gray')

        ''

        st.line_chart(
            filtered_gdp_df,
            x='Year',
            y='GDP',
            color='Country Code',
        )

        ''
        ''


        first_year = gdp_df[gdp_df['Year'] == from_year]
        last_year = gdp_df[gdp_df['Year'] == to_year]

        st.header(f'PIB en {to_year}', divider='gray')

        ''

        cols = st.columns(4)

        for i, country in enumerate(selected_countries):
            col = cols[i % len(cols)]

            with col:
                first_gdp = first_year[gdp_df['Country Code'] == country]['GDP'].iat[0] / 1000000000
                last_gdp = last_year[gdp_df['Country Code'] == country]['GDP'].iat[0] / 1000000000

                if math.isnan(first_gdp):
                    growth = 'n/a'
                    delta_color = 'off'
                else:
                    growth = f'{last_gdp / first_gdp:,.2f}x'
                    delta_color = 'normal'

                st.metric(
                    label=f'{country} GDP',
                    value=f'{last_gdp:,.0f}B',
                    delta=growth,
                    delta_color=delta_color
                )
 

        # Afficher le graphique
        st.plotly_chart(fig)

        # Tabs content
        #st.write(f"# Top 10 des actualités du jour")
        
       

        file_path = 'export-12.xlsx'
        display_excel_file(file_path)
        
        display_finnhub_news()

        # Affichage du contenu HTML dans Streamlit
        #st.markdown(scrolling_logos, unsafe_allow_html=True)

    if app_mode == 'Recherche':
        ticker = st.text_input('Entrez le symbole du ticker (par ex. AAPL)', 'AAPL')
        start_date = st.date_input('Date de début', dt.date(2020, 1, 1))
        end_date = st.date_input('Date de fin', dt.date.today())
        forecast_days = st.number_input("Nombre de jours à prédire", min_value=1, max_value=30, value=7)
        period = st.selectbox("Période d'analyse:", options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"])


        with st.spinner('Loading...'):
            if st.button('Télécharger les données'):
                data, stock_name, info = download_stock_data(ticker, start_date, end_date)
                st.write(f"### {stock_name} ({ticker})")
                # Lien dynamique vers ValueInvesting basé sur le ticker
                st.write(f"[Voir l'évaluation WACC sur ValueInvesting pour {ticker}](https://valueinvesting.io/{ticker}/valuation/wacc)")

                display_company_info(info)

                from scipy.optimize import curve_fit

                # Télécharger les données historiques pour un ticker donné
                data = yf.download(ticker, start="2024-01-01", end="2024-08-30")

                # Calculer les rendements quotidiens
                data['Returns'] = data['Adj Close'].pct_change()

                # Modèle hypothétique basé sur KAN
                def kan_model(t, P0, k, sigma0):
                    # Modéliser la volatilité comme une fonction de t (temps), P0 (prix), et sigma0 (volatilité initiale)
                    # On simplifie en supposant P(t) = P0 * t et sigma(t) = sigma0 * sqrt(t)
                    return P0 * t + k * sigma0 * np.sqrt(t)

                # Préparer les données pour l'ajustement du modèle
                data = data.dropna()
                t = np.arange(len(data))
                sigma_t = data['Returns'].rolling(window=20).std().values * np.sqrt(252)  # Volatilité annualisée

                # Nettoyer les données en enlevant les valeurs NaNs ou infinies
                mask = np.isfinite(sigma_t) & np.isfinite(t)
                t = t[mask]
                sigma_t = sigma_t[mask]

                # Ajuster le modèle KAN aux données
                popt, _ = curve_fit(kan_model, t, sigma_t, p0=[1, 0.5, 0.02])

                # Prédire les volatilités à l'aide du modèle ajusté
                predicted_volatility = kan_model(t, *popt)

                # Créer un graphique avec Plotly
                fig = go.Figure()

                # Ajouter la volatilité observée
                fig.add_trace(go.Scatter(x=data.index[mask],
                                        y=sigma_t,
                                        mode='lines',
                                        name="Volatilité Observée",
                                        line=dict(color='blue')))

                # Ajouter la volatilité prédite
                fig.add_trace(go.Scatter(x=data.index[mask],
                                        y=predicted_volatility,
                                        mode='lines',
                                        name="Volatilité Prédite (Modèle KAN)",
                                        line=dict(color='red', dash='dash')))

                # Mettre à jour la mise en page du graphique
                fig.update_layout(title=f"Modélisation de la Volatilité avec le Modèle KAN pour {ticker}",
                                xaxis_title="Date",
                                yaxis_title="Volatilité",
                                plot_bgcolor='white',
                                xaxis_rangeslider_visible=False)
                
                # Add annotation in the bottom left
                fig.add_annotation(
                    text='StockGenius',
                    xref='paper', yref='paper',
                    x=0.01, y=0.01,
                    showarrow=False,
                    font=dict(
                        size=10,
                        color='black'
                    ),
                    align='left',
                    opacity=0.5
                )

                # Afficher le graphique
                st.plotly_chart(fig)
                
                # Plot linear regression
                
                plot_linear_regression(data)
                predicted_price, win_rate = predict_stock_prices_advanced(ticker, forecast_days)
                st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")

                
                
                
                plot_prediction(ticker, forecast_days, predicted_price, win_rate)
        
                st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")

                


                st.write(f"# Section Finance")
                

                if ticker:
                    display_financial_summary(ticker)
                    import yfinance as yf
                    import streamlit as st
                    import plotly.express as px
                    import pandas as pd

                    # Fonction pour télécharger les états financiers d'une entreprise
                    def download_financials(ticker):
                        stock = yf.Ticker(ticker)
                        financials = stock.financials.T  # Compte de résultat
                        balance_sheet = stock.balance_sheet.T  # Bilan
                        cashflow = stock.cashflow.T  # Flux de trésorerie
                        return financials, balance_sheet, cashflow

                    # Fonction pour formater les données en millions de dollars
                    def format_in_millions(df):
                        return df.applymap(lambda x: x / 1e6 if pd.notnull(x) else x)

                    # Fonction pour afficher les états financiers sous forme de graphiques en tarte
                    def plot_pie_chart(financials, title, colors):
                        # Agréger les données par colonne et garder les 5 plus gros éléments
                        financials_sum = financials.sum(axis=0)
                        top_5 = financials_sum.nlargest(5)  # Sélectionner les 5 plus gros éléments
                        
                        fig = px.pie(
                            values=top_5.values,
                            names=top_5.index,
                            title=title,
                            color_discrete_sequence=colors
                        )
                        fig.update_traces(textinfo='percent+label')
                        
                        # Mise à jour du layout pour séparer la légende du graphique
                        fig.update_layout(
                            title_font=dict(size=20, color='White', family="Arial, sans-serif"),
                            legend=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.2,  # Positionner la légende bien en dessous du graphique
                                xanchor="center",
                                x=0.5,
                                font=dict(size=12),
                                bgcolor='rgba(255, 255, 255, 0.6)',
                                bordercolor="White",
                                borderwidth=1
                            ),
                            margin=dict(l=0, r=0, t=40, b=100)  # Ajouter de l'espace pour éviter le chevauchement
                        )
                        st.plotly_chart(fig, use_container_width=True)

                

                    

                    # Télécharger les états financiers
                    financials, balance_sheet, cashflow = download_financials(ticker)

                    # Palette de couleurs pour les graphiques
                    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']

                    
                    # Vérification de la présence des données pour chaque tableau avant affichage
                    if not financials.empty:
                        st.header("Compte de Résultat")
                        financials_in_millions = format_in_millions(financials)
                        plot_pie_chart(financials_in_millions, "StockGenius", colors)

                    if not balance_sheet.empty:
                        st.header("Bilan")
                        balance_sheet_in_millions = format_in_millions(balance_sheet)
                        plot_pie_chart(balance_sheet_in_millions, "StockGenius", colors)

                    if not cashflow.empty:
                        st.header("Flux de Trésorerie")
                        cashflow_in_millions = format_in_millions(cashflow)
                        plot_pie_chart(cashflow_in_millions, "StockGenius", colors)
                    
                    # Calculer automatiquement le changement de prix
                    change = get_price_change(ticker, period=period)

                    
            
                    if change is not None:
                        # Afficher la jauge
                        st.write(f"# Sentiments des invesstisseurs pour {ticker} ")
                        fig = create_gauge(change)
                        st.plotly_chart(fig)

                        # Ajout de l'annotation "StockGenius" en bas à gauche
                        st.markdown(
                            "<div style='text-align: left; color: #888; font-size: 10pt; margin-top: 10px;'>StockGenius</div>",
                            unsafe_allow_html=True
                        )

                        # Afficher le changement en pourcentage
                        st.write(f"Changement de prix pour {ticker} sur la période {period} : {change * 100:.2f}%")
                        # Récupérer les actualités
                        news_data = get_stock_news(ticker)

                        # Si des données sont retournées
                        if news_data:
                            st.write(f"# Actualités pour {ticker.upper()}")

                            # Limiter aux 5 premières actualités
                            limited_news = news_data[:5]

                            # Afficher les actualités
                            for index, row in enumerate(limited_news):
                                st.write(f"**{index + 1}. Titre :** {row['headline']}")
                                st.write(f"**Source :** {row['source']}")
                                st.write(f"[Lire l'article]({row['url']})")
                                st.write("---")
                        else:
                            st.info(f"Aucune actualité trouvée pour {ticker.upper()}.")
                        

                        
                else:
                    st.write(f"Aucune donnée historique disponible pour le ticker {ticker} sur la période {period}.")
                    
    if app_mode == 'Prédictions':
        ticker = st.text_input('Entrez le symbole du ticker (par ex. AAPL)', 'AAPL')
        start_date = st.date_input('Date de début', dt.date(2000, 1, 1))
        end_date = st.date_input('Date de fin', dt.date.today())
        num_simulations = st.number_input('Nombre de simulations', value=100, min_value=10, max_value=100000)
        num_days = st.number_input('Nombre de jours de prédiction', value=252, min_value=1, max_value=365)
        with st.spinner('Loading...'):

            if st.button('Lancer la simulation'):
                data, _, _ = download_stock_data(ticker, start_date, end_date)
                simulation_df, last_price, mean_price = monte_carlo_simulation(data, num_simulations, num_days)

                st.subheader('Résultat des Simulations')
                st.write(f"Prix actuel: ${last_price:.2f}")
                st.write(f"Prix moyen prédit: ${mean_price:.2f}")

                # Plot results
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(simulation_df)
                ax.axhline(y=last_price, color='r', linestyle='--', label=f'Prix actuel: ${last_price:.2f}')
                ax.axhline(y=mean_price, color='g', linestyle='--', label=f'Prix moyen prédit: ${mean_price:.2f}')
                ax.set_title(f'Simulation Monte Carlo pour {ticker}')
                ax.set_xlabel('Jour')
                ax.set_ylabel('Prix')
                ax.legend()
                plt.figtext(0.01, 0.01, 'StockGenius', fontsize=12, color='gray')
                st.pyplot(fig)
                st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")

    if app_mode == 'Options':
        ticker = st.text_input('Entrez le symbole du ticker (par ex. AAPL)', 'AAPL')
        expiry_date = st.selectbox('Date d\'expiration', st.session_state.available_expirations)
        forecast_days = st.number_input("Nombre de jours à prédire", min_value=1, max_value=30, value=7)

        if st.button('Mettre à jour les dates d\'expiration'):
            _, _, _, _, _, available_expirations = fetch_option_data(ticker, expiry_date)
            st.session_state.available_expirations = available_expirations

        with st.spinner('Loading...'):
            if st.button('Afficher les options'):
                S, strikes, market_prices, T, r, _ = fetch_option_data(ticker, expiry_date)
                if S is not None:
                    ivs = [implied_volatility(S, K, T, r, P) for K, P in zip(strikes, market_prices)]
                    #st.write("# Données sur les options")
                    option_data = pd.DataFrame({
                        'Strike': strikes,
                        'Prix du marché': market_prices,
                        'Volatilité implicite': ivs
                    })
                    #display_option_data(strikes, market_prices, ivs)
                    plot_volatility_surface(ticker, expiry_date, forecast_days)
                    st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")


                    # Plot implied volatility vs strike price
                    fig_volatility = go.Figure()
                    fig_volatility.add_trace(go.Scatter(
                        x=option_data['Strike'],
                        y=option_data['Volatilité implicite'],
                        mode='lines+markers',
                        name='Volatilité implicite',
                        line=dict(color='blue')
                    ))
                    
                    fig_volatility.update_layout(
                        title='Volatilité Implicite en Fonction du Prix d\'Exercice',
                        xaxis_title='Prix d\'exercice',
                        yaxis_title='Volatilité implicite',
                        plot_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig_volatility)
                    # Ajout de l'annotation "StockGenius" en bas à gauche
                    st.markdown(
                        "<div style='text-align: left; color: #888; font-size: 10pt; margin-top: 10px;'>StockGenius</div>",
                        unsafe_allow_html=True
                    )
                    st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")


                    # Plot implied volatility vs market price
                    fig_market_price = go.Figure()
                    fig_market_price.add_trace(go.Scatter(
                        x=option_data['Prix du marché'],
                        y=option_data['Volatilité implicite'],
                        mode='markers',
                        name='Volatilité implicite',
                        marker=dict(color='red')
                    ))
                    fig_market_price.update_layout(
                        title='Volatilité Implicite en Fonction du Prix du Marché',
                        xaxis_title='Prix du marché',
                        yaxis_title='Volatilité implicite',
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig_market_price)
                    # Ajout de l'annotation "StockGenius" en bas à gauche
                    st.markdown(
                        "<div style='text-align: left; color: #888; font-size: 10pt; margin-top: 10px;'>StockGenius</div>",
                        unsafe_allow_html=True
                    )
                    st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")

    if app_mode == 'Prévision Économique':
        country = st.selectbox('Choisissez un pays', ['États-Unis', 'Canada'])
        api_key = st.text_input('API', 'rBnQyygXVXNxBvqqFBY1')

        if api_key and st.button('Prévoir'):
            data_gdp = None
            data_unemployment = None
            data_inflation = None

            if country == 'États-Unis':
                data_gdp = quandl.get("FRED/GDP", authtoken=api_key)
                data_unemployment = quandl.get("FRED/UNRATE", authtoken=api_key)
                data_inflation = quandl.get("FRED/CPIAUCSL", authtoken=api_key)
                if data_gdp is not None:

                    # Données
                    pays = ['États-Unis', 'Allemagne', 'Italie', 'France', 'Autres pays de l\'UE']
                    dette = [30.7, 2.6, 2.8, 3.1, 5.2]
                    groupes = ['États-Unis', 'UE-27', 'UE-27', 'UE-27', 'UE-27']

                    # Création du graphique en barres
                    fig = go.Figure()

                    # Barres pour chaque pays
                    colors = ['#800000', '#ff6666', '#ff4d4d', '#ff1a1a', '#cc0000']  # Couleurs similaires au graphique original

                    for i in range(len(pays)):
                        fig.add_trace(go.Bar(
                            x=[dette[i]],
                            y=[pays[i]],
                            name=pays[i],
                            orientation='h',
                            marker=dict(color=colors[i]),
                            hovertemplate=f'{pays[i]}: {dette[i]} billions d\'euros<extra></extra>'
                        ))

                    # Mise en page
                    fig.update_layout(
                        title="In Debt We Trust",
                        title_font_size=24,
                        title_x=0.5,
                        xaxis_title="Dette publique (en billions d'euros)",
                        yaxis_title="",
                        yaxis=dict(categoryorder='total ascending'),  # Trie les pays par montant
                        xaxis=dict(showgrid=False),
                        plot_bgcolor='white',
                        showlegend=False,
                        margin=dict(l=100, r=20, t=60, b=40)
                    )

                    # Affichage du graphique dans Streamlit
                    st.plotly_chart(fig)

                    import plotly.express as px
                    # Exemple de données similaires à celles de l'image
                    data = {
                        'Pays': ['Tous les autres', 'Japon', 'Chine', 'Royaume-Uni', 'Luxembourg', 'Canada', 'Irlande', 'Îles Caïmans',
                                'Belgique', 'Suisse', 'France', 'Taiwan', 'Inde', 'Hong Kong', 'Brésil', 'Singapour', 'Norvège',
                                'Arabie Saoudite', 'Corée du Sud', 'Allemagne', 'Bermudes'],
                        'Milliards de dollars': [1385.3, 1153.1, 797.7, 753.5, 376.5, 339.8, 319.7, 318.5, 293.1, 283.1, 267.5, 255.9, 
                                                236.1, 228.5, 226.8, 204.7, 144.9, 127.5, 118.6, 106.5, 86.2]
                    }

                    # Création du DataFrame
                    df = pd.DataFrame(data)

                    # Création du graphique interactif
                    fig = px.bar(df, 
                                x='Milliards de dollars', 
                                y='Pays', 
                                orientation='h', 
                                title="Principaux détenteurs étrangers de titres du Trésor américain en janvier 2024",
                                labels={'Milliards de dollars': 'en milliards de dollars', 'Pays': ''},
                                text='Milliards de dollars')

                    # Ajustement de la mise en page
                    fig.update_layout(
                        yaxis=dict(categoryorder='total ascending'),  # Trie les pays par montant
                        xaxis_title='En milliards de dollars',
                        yaxis_title='',
                        title={
                            'text': "Principaux détenteurs étrangers de titres du Trésor américain en janvier 2024",
                            'y': 0.95,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        font=dict(
                            size=12
                        )
                    )

                    # Affichage du graphique dans Streamlit
                    st.plotly_chart(fig)

                    st.subheader("Produit Intérieur Brut (PIB)")
                
                    # Calcul des moyennes mobiles
                    data_gdp['30_Day_MA'] = data_gdp['Value'].rolling(window=30).mean()
                    data_gdp['100_Day_MA'] = data_gdp['Value'].rolling(window=100).mean()
                    
                    # Tracer les données avec les moyennes mobiles
                    fig_gdp = go.Figure()
                    fig_gdp.add_trace(go.Scatter(
                        x=data_gdp.index,
                        y=data_gdp['Value'],
                        mode='lines',
                        name='PIB',
                        line=dict(color='blue')
                    ))
                    fig_gdp.add_trace(go.Scatter(
                        x=data_gdp.index,
                        y=data_gdp['30_Day_MA'],
                        mode='lines',
                        name='Moyenne Mobile 30 Jours',
                        line=dict(color='orange', dash='dash')
                    ))
                    fig_gdp.add_trace(go.Scatter(
                        x=data_gdp.index,
                        y=data_gdp['100_Day_MA'],
                        mode='lines',
                        name='Moyenne Mobile 100 Jours',
                        line=dict(color='green', dash='dash')
                    ))
                    fig_gdp.update_layout(
                        title='Produit Intérieur Brut avec Moyennes Mobiles',
                        xaxis_title='Date',
                        yaxis_title='PIB',
                        plot_bgcolor='white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_gdp)
                    
                    # Analyse des statistiques
                    last_value = data_gdp['Value'].iloc[-1]
                    max_value = data_gdp['Value'].max()
                    min_value = data_gdp['Value'].min()
                    
                    pct_change = data_gdp['Value'].pct_change()
                    growth_mean = pct_change.mean() * 100
                    annual_growth = (data_gdp['Value'].iloc[-1] / data_gdp['Value'].iloc[-60] - 1) * 100 if len(data_gdp) > 60 else float('nan')
                    volatility = pct_change.std() * 100
                    negative_growth_count = (pct_change < 0).sum()
                    annual_growth_change = data_gdp['Value'].pct_change(periods=4).mean() * 100
                    trend_last_value = data_gdp['Value'].rolling(window=12).mean().iloc[-1]
                    
                    st.write("**Analyse historique :**")
                    st.write(f"- Valeur actuelle : {last_value:,.2f}")
                    st.write(f"- Valeur maximale sur la période : {max_value:,.2f}")
                    st.write(f"- Valeur minimale sur la période : {min_value:,.2f}")
                    st.write(f"- Croissance annuelle moyenne : {growth_mean:.2f}%")
                    st.write(f"- Taux de croissance du PIB sur les 5 dernières années : {annual_growth:.2f}%")
                    st.write(f"- Variabilité du PIB (écart type) : {volatility:.2f}%")
                    st.write(f"- Nombre de périodes de croissance négative : {negative_growth_count}")
                    st.write(f"- Taux de croissance du PIB en glissement annuel : {annual_growth_change:.2f}%")
                    st.write(f"- Tendances observées : {trend_last_value:,.2f}")
                    
                    # Exemple simple de prévision avec régression linéaire
                    from sklearn.linear_model import LinearRegression
                    data_gdp_reset = data_gdp.reset_index()
                    data_gdp_reset['Date_Ordinal'] = pd.to_datetime(data_gdp_reset['Date']).map(pd.Timestamp.toordinal)
                    X = data_gdp_reset[['Date_Ordinal']]
                    y = data_gdp_reset['Value']
                    model = LinearRegression()
                    model.fit(X, y)
                    predicted_value = model.predict([[data_gdp_reset['Date_Ordinal'].iloc[-1]]])[0]
                    st.write(f"- Modèle de prévision simple : {predicted_value:,.2f}")

                    if data_unemployment is not None:
                        st.subheader("Taux de Chômage")
                        
                        # Calcul des moyennes mobiles
                        data_unemployment['30_Day_MA'] = data_unemployment['Value'].rolling(window=30).mean()
                        data_unemployment['100_Day_MA'] = data_unemployment['Value'].rolling(window=100).mean()
                        
                        # Tracer les données avec les moyennes mobiles
                        fig_unemployment = go.Figure()
                        fig_unemployment.add_trace(go.Scatter(
                            x=data_unemployment.index,
                            y=data_unemployment['Value'],
                            mode='lines',
                            name='Taux de Chômage',
                            line=dict(color='blue')
                        ))
                        fig_unemployment.add_trace(go.Scatter(
                            x=data_unemployment.index,
                            y=data_unemployment['30_Day_MA'],
                            mode='lines',
                            name='Moyenne Mobile 30 Jours',
                            line=dict(color='orange', dash='dash')
                        ))
                        fig_unemployment.add_trace(go.Scatter(
                            x=data_unemployment.index,
                            y=data_unemployment['100_Day_MA'],
                            mode='lines',
                            name='Moyenne Mobile 100 Jours',
                            line=dict(color='green', dash='dash')
                        ))
                        fig_unemployment.update_layout(
                            title='Taux de Chômage avec Moyennes Mobiles',
                            xaxis_title='Date',
                            yaxis_title='Taux de Chômage',
                            plot_bgcolor='white',
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_unemployment)
                        
                        # Analyse des statistiques
                        last_value = data_unemployment['Value'].iloc[-1]
                        max_value = data_unemployment['Value'].max()
                        min_value = data_unemployment['Value'].min()
                        
                        pct_change = data_unemployment['Value'].pct_change()
                        growth_mean = pct_change.mean() * 100
                        annual_growth = (data_unemployment['Value'].iloc[-1] / data_unemployment['Value'].iloc[-60] - 1) * 100 if len(data_unemployment) > 60 else float('nan')
                        volatility = pct_change.std() * 100
                        negative_growth_count = (pct_change < 0).sum()
                        annual_growth_change = data_unemployment['Value'].pct_change(periods=4).mean() * 100
                        trend_last_value = data_unemployment['Value'].rolling(window=12).mean().iloc[-1]
                        
                        st.write("**Analyse historique :**")
                        st.write(f"- Taux actuel : {last_value:.2f}%")
                        st.write(f"- Taux maximal sur la période : {max_value:.2f}%")
                        st.write(f"- Taux minimal sur la période : {min_value:.2f}%")
                        st.write(f"- Variation annuelle moyenne : {growth_mean:.2f}%")
                        st.write(f"- Nombre de mois avec des augmentations du taux de chômage : {(data_unemployment.pct_change() > 0).sum()}")
                        st.write(f"- Nombre de mois avec des baisses du taux de chômage : {(data_unemployment.pct_change() < 0).sum()}")
                        st.write(f"- Taux de chômage sur les 5 dernières années : {annual_growth:.2f}%")
                        st.write(f"- Écart type du taux de chômage : {volatility:.2f}%")
                        
                        # Exemple simple de prévision avec régression linéaire
                        model = LinearRegression()
                        data_unemployment_reset = data_unemployment.reset_index()
                        data_unemployment_reset['Date_Ordinal'] = pd.to_datetime(data_unemployment_reset['Date']).map(pd.Timestamp.toordinal)
                        X = data_unemployment_reset[['Date_Ordinal']]
                        y = data_unemployment_reset['Value']
                        model.fit(X, y)
                        predicted_value = model.predict([[data_unemployment_reset['Date_Ordinal'].iloc[-1]]])[0]
                        st.write(f"- Modèle de prévision simple : {predicted_value:.2f}")

                        if data_inflation is not None:
                            st.subheader("Inflation")
                            
                            # Calcul des moyennes mobiles
                            data_inflation['30_Day_MA'] = data_inflation['Value'].rolling(window=30).mean()
                            data_inflation['100_Day_MA'] = data_inflation['Value'].rolling(window=100).mean()
                            
                            # Tracer les données avec les moyennes mobiles
                            fig_inflation = go.Figure()
                            fig_inflation.add_trace(go.Scatter(
                                x=data_inflation.index,
                                y=data_inflation['Value'],
                                mode='lines',
                                name='Inflation',
                                line=dict(color='blue')
                            ))
                            fig_inflation.add_trace(go.Scatter(
                                x=data_inflation.index,
                                y=data_inflation['30_Day_MA'],
                                mode='lines',
                                name='Moyenne Mobile 30 Jours',
                                line=dict(color='orange', dash='dash')
                            ))
                            fig_inflation.add_trace(go.Scatter(
                                x=data_inflation.index,
                                y=data_inflation['100_Day_MA'],
                                mode='lines',
                                name='Moyenne Mobile 100 Jours',
                                line=dict(color='green', dash='dash')
                            ))
                            fig_inflation.update_layout(
                                title='Inflation avec Moyennes Mobiles',
                                xaxis_title='Date',
                                yaxis_title='Inflation',
                                plot_bgcolor='white',
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_inflation)
                            
                            # Analyse des statistiques
                            last_value = data_inflation['Value'].iloc[-1]
                            max_value = data_inflation['Value'].max()
                            min_value = data_inflation['Value'].min()
                            
                            pct_change = data_inflation['Value'].pct_change()
                            growth_mean = pct_change.mean() * 100
                            annual_growth = (data_inflation['Value'].iloc[-1] / data_inflation['Value'].iloc[-60] - 1) * 100 if len(data_inflation) > 60 else float('nan')
                            volatility = pct_change.std() * 100
                            negative_growth_count = (pct_change < 0).sum()
                            annual_growth_change = data_inflation['Value'].pct_change(periods=4).mean() * 100
                            trend_last_value = data_inflation['Value'].rolling(window=12).mean().iloc[-1]
                            
                            st.write("**Analyse historique :**")
                            st.write(f"- Valeur actuelle : {last_value:.2f}")
                            st.write(f"- Valeur maximale sur la période : {max_value:.2f}")
                            st.write(f"- Valeur minimale sur la période : {min_value:.2f}")
                            st.write(f"- Croissance annuelle moyenne : {growth_mean:.2f}%")
                            st.write(f"- Taux d'inflation sur les 5 dernières années : {annual_growth:.2f}%")
                            st.write(f"- Variabilité de l'inflation (écart type) : {volatility:.2f}%")
                            st.write(f"- Nombre de mois avec une inflation négative : {negative_growth_count}")
                            st.write(f"- Croissance annuelle moyenne de l'inflation : {annual_growth_change:.2f}%")
                            st.write(f"- Tendances observées : {trend_last_value:.2f}")
                            
                            # Exemple simple de prévision avec régression linéaire
                            model = LinearRegression()
                            data_inflation_reset = data_inflation.reset_index()
                            data_inflation_reset['Date_Ordinal'] = pd.to_datetime(data_inflation_reset['Date']).map(pd.Timestamp.toordinal)
                            X = data_inflation_reset[['Date_Ordinal']]
                            y = data_inflation_reset['Value']
                            model.fit(X, y)
                            predicted_value = model.predict([[data_inflation_reset['Date_Ordinal'].iloc[-1]]])[0]
                            st.write(f"- Modèle de prévision simple : {predicted_value:.2f}")
                        
                            

        else:
            st.error("Impossible de récupérer les données. Vérifiez votre clé API et réessayez.")

    if app_mode == 'Obligation':
        # Télécharger les données d'obligation
        ticker = st.text_input('Entrez le ticker de l\'obligation (par ex. TLT)', 'TLT')
        start_date = st.date_input('Date de début', dt.date(2022, 1, 1))
        end_date = st.date_input('Date de fin', dt.date.today())
        forecast_days = st.number_input("Nombre de jours à prédire", min_value=1, max_value=30, value=7)

        if st.button('Télécharger les données'):
            with st.spinner('Loading...'):
                data, bond_name, info = download_bond_data(ticker, start_date, end_date)
                st.write(f"### {bond_name} ({ticker})")
                st.line_chart(data)

                # Ajouter la régression linéaire
                data_df = data.to_frame(name='Close')
                plot_linear_regression(data_df)
                st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")


                predicted_price, win_rate = predict_stock_prices_advanced(ticker, forecast_days)
                st.write(f"# Machine Learning Prévision")
                st.write(f"Prix prédit: ${predicted_price[0]:.2f}")
                st.write(f"Taux de réussite: {win_rate:.2%}")
                plot_prediction(ticker, forecast_days, predicted_price, win_rate)
                st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")
             
    if app_mode == 'Gestion des Actifs':
        # Entrée de tickers sous forme de chaîne de caractères
        
        tickers_input = st.text_input("Entrez les tickers (séparés par des virgules)", "BFH, SDE.TO, AAPL")
        ticker = [ticker.strip() for ticker in tickers_input.split(',')]  # Convertir en liste de tickers
        start = st.date_input('Date de début', dt.date(2022, 1, 1))
        end = st.date_input('Date de fin', dt.date.today())
        


        prices_df = get_price_history(' '.join(ticker), sdate=start, edate=end)
        returns_df = prices_df.pct_change()[1:]
        st.subheader("Performance des Actions")
        plot_performance(prices_df)
        st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")


        st.subheader("Frontière Efficiente")
        plot_efficient_frontier(prices_df)
        st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")

    if app_mode == "Screener":
        file_path = 'export-12'
        display_excel_file(file_path)
        # Ajout de l'annotation "StockGenius" en bas à gauche
        st.markdown(
            "<div style='text-align: left; color: #888; font-size: 10pt; margin-top: 10px;'>StockGenius</div>",
            unsafe_allow_html=True
        )
        
    if app_mode == "Sources":

        
        
        
        st.title("Mentions Légales")
    
        st.markdown("""
        ## 1. Introduction
        Les présentes mentions légales régissent l'utilisation de ce site web (le "Site"). En accédant au Site, vous acceptez de respecter ces conditions. Si vous n'acceptez pas ces conditions, veuillez ne pas utiliser le Site.

        ## 2. Nature des Informations
        Les informations présentées sur le Site sont fournies à titre théorique et éducatif uniquement. Elles ne constituent pas des conseils financiers, d'investissement, ou juridiques. Nous ne garantissons pas l'exactitude, la complétude ou l'actualité des informations fournies.

        ## 3. Aucune Responsabilité
        L'utilisation des informations disponibles sur le Site est à vos propres risques. En aucun cas, le Site, ses propriétaires, ou ses employés ne pourront être tenus responsables des pertes ou dommages directs, indirects, spéciaux ou consécutifs résultant de l'utilisation des informations ou de l'incapacité d'utiliser le Site.

        ## 4. Avertissement concernant la Théorie Financière
        Les simulations, prévisions, et analyses présentées sur le Site sont basées sur des modèles théoriques et ne reflètent pas nécessairement la réalité du marché. Les résultats passés ne garantissent pas les résultats futurs.

        ## 5. Limitation de Responsabilité Légale
        Le Site ne peut être tenu responsable de toute décision prise sur la base des informations présentées. Les utilisateurs sont encouragés à consulter un professionnel qualifié avant de prendre des décisions financières ou d'investissement.

        ## 6. Modifications des Mentions Légales
        Nous nous réservons le droit de modifier ces mentions légales à tout moment. Les utilisateurs sont invités à consulter régulièrement cette page pour prendre connaissance des éventuelles modifications.

        ## 7. Droit Applicable
        Les présentes mentions légales sont régies par les lois de la province de [Votre Province], Canada. Tout litige découlant de l'utilisation du Site sera soumis à la compétence exclusive des tribunaux de cette province.

        ## 8. Contact
        Pour toute question concernant ces mentions légales, veuillez nous contacter à [adresse e-mail de contact].
        """)
        
        
        st.write("""
        ### Importance des Sources de Qualité
        Avoir des sources de qualité est crucial pour obtenir des informations fiables et précises, particulièrement dans le domaine de l'investissement. Les sources de qualité fournissent des données vérifiées et des analyses approfondies, ce qui aide à prendre des décisions éclairées et à éviter les pièges des informations erronées ou biaisées.
        """)

        st.write("Voici quelques sources de qualité pour vos recherches :")

        st.write("[Investopedia](https://www.investopedia.com)")
        st.write("[Banque centrale américaine (Federal Reserve)](https://www.federalreserve.gov)")
        st.write("[Banque du Canada (Bank of Canada)](https://www.bankofcanada.ca)")
        st.write("[Questrade](https://www.questrade.com)")
        st.write("[Seeking Alpha](https://seekingalpha.com)")
        st.write("[Zacks](https://www.zacks.com)")
        st.write("[Sedar](https://www.sedarplus.ca)")

        # Code HTML et CSS pour le défilement infini des logos
        scrolling_logos = """
        <style>
        @keyframes scroll {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
        }

        .scrolling-container {
        width: 100%;
        overflow: hidden;
        position: relative;
        background-color: #f8f9fa; /* Couleur de fond, optionnelle */
        padding: 10px 0;
        }

        .scrolling-logos {
        display: flex;
        justify-content: center;
        align-items: center;
        animation: scroll 40s linear infinite; /* 40s pour une vitesse plus lente */
        }

        .scrolling-logos img {
        height: 50px; /* Taille des logos */
        margin: 0 20px; /* Espacement entre les logos */
        }
        </style>

        <div class="scrolling-container">
        <div class="scrolling-logos">
            <img src="https://upload.wikimedia.org/wikipedia/commons/1/1a/Seal_of_the_United_States_Federal_Reserve_System.svg" alt="FOMC">
            <img src="https://upload.wikimedia.org/wikipedia/commons/8/87/NASDAQ_Logo.svg" alt="NASDAQ">
            <img src="https://upload.wikimedia.org/wikipedia/commons/b/be/NYSE_Logo.svg" alt="NYSE">
            <img src="https://upload.wikimedia.org/wikipedia/commons/b/bd/TSX_Logo.svg" alt="TSX">
            <img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Bloomberg_logo.svg" alt="Bloomberg">
            <img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Yahoo%21_%282019%29.svg" alt="Yahoo">
            <img src="https://upload.wikimedia.org/wikipedia/en/0/0c/MarketWatch_Logo.svg" alt="MarketWatch">
            <img src="https://store-images.s-microsoft.com/image/apps.18771.7c58e601-ae93-41a7-8654-f7243c835eee.097f998a-684c-49c1-b8de-e12877c70b81.a4ef990c-549c-488b-a2c7-aa6b455b672f.png" alt="ValueInvesting.io">
            <img src="https://1000logos.net/wp-content/uploads/2023/11/Forex-Factory-Logo.jpg" alt="Forex Factory">
            <img src="https://upload.wikimedia.org/wikipedia/commons/e/e3/CNBC_logo.svg" alt="CNBC">
            <img src="https://upload.wikimedia.org/wikipedia/commons/3/37/Investopedia_Logo.svg" alt="INVESTOPEDIA">
            <img src="https://upload.wikimedia.org/wikipedia/fr/d/d1/Les_affaires_%28logo%29.png" alt="LESAFFAIRES">
            <img src="https://www.finance-investissement.com/wp-content/uploads/sites/2/2018/01/fi-logo.svg" alt="Finance">
        </div>
        </div>
        """

        # Affichage du contenu HTML dans Streamlit
        st.markdown(scrolling_logos, unsafe_allow_html=True)

        st.markdown("### Nos partenaires d'informations")

    if app_mode == "Future":
        st.title("Analyse des Futures")

        # Introduction
        st.markdown("""
        ### Qu'est-ce qu'un contrat à terme (Future) ?
        Les contrats à terme sont des accords financiers pour acheter ou vendre un actif à un prix prédéterminé à une date spécifique dans le futur. 
        Ils sont utilisés par les investisseurs pour spéculer sur la direction future des prix ou pour se couvrir contre des mouvements de prix défavorables.
        Les futures sont négociés sur divers actifs tels que les indices boursiers, les matières premières, les taux d'intérêt, et plus encore.
        """)

        # Sélection du ticker et des dates
        ticker = st.selectbox('Sélectionnez un Future:', ['ES=F', 'CL=F', 'GC=F', 'ZB=F', 'NQ=F'])
        start_date = st.date_input('Date de début', value=pd.to_datetime('2022-01-01'))
        end_date = st.date_input('Date de fin', value=pd.to_datetime('today'))
        forecast_days = st.number_input("Nombre de jours à prédire", min_value=1, max_value=30, value=7)

        # Téléchargement des données
        futures_data = download_futures_data(ticker, start_date, end_date)

        # Affichage du graphique
        plot_futures_data(futures_data, ticker)
        
        st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")

        predicted_price, win_rate = predict_stock_prices_advanced(ticker, forecast_days)
        st.write(f"# Machine Learning Prévision")
        st.write(f"Prix prédit: ${predicted_price[0]:.2f}")
        st.write(f"Taux de réussite: {win_rate:.2%}")
        plot_prediction(ticker, forecast_days, predicted_price, win_rate)
        st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")

    if app_mode == "FOREX":
        ticker = st.selectbox('Sélectionnez un Future:', ['USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X',
                                                        'USDCHF=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'EURCHF=X'])
        
        start_date = st.date_input('Date de début', value=pd.to_datetime('2022-01-01'))
        end_date = st.date_input('Date de fin', value=pd.to_datetime('today'))
        forecast_days = st.number_input("Nombre de jours à prédire", min_value=1, max_value=30, value=7)

        if ticker:
            data = get_forex_data(ticker)

            if data:
                st.subheader(f'Informations pour {ticker}')
                st.write(f"**Prix de clôture :** {data['Close']}")
                st.write(f"**Prix d'ouverture :** {data['Open']}")
                st.write(f"**Prix le plus haut :** {data['High']}")
                st.write(f"**Prix le plus bas :** {data['Low']}")
                st.write(f"**Volume :** {data['Volume']}")
                st.write(f"**Date de mise à jour :** {data['Date']}")
            
                
                predicted_price, win_rate = predict_stock_prices_advanced(ticker, forecast_days)

                st.write(f"# Machine Learning Prévision")
                st.write(f"Prix prédit: ${predicted_price[0]:.2f}")
                st.write(f"Taux de réussite: {win_rate:.2%}")

                plot_prediction(ticker, forecast_days, predicted_price, win_rate)
                st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")
   
    if app_mode == "Backtesting":

        import time
        from bokeh.models import DatetimeTickFormatter

        # Customize your plot before calling bt.plot()
        def custom_plot(bt):
            p = bt.plot()
            p.xaxis.formatter = DatetimeTickFormatter(days="%d %b")
            return p

        st.write(f"# Stratégies de Backtesting - Trading")
        st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")


        strategy_box = st.selectbox('Sélectionnez une Stratégie GeniusStock:', ['RSI, ATR, TRIX & MOMENTUM STRAT', 'SMA CROSS STRAT', 'RSI STRAT'])

        ticker = st.text_input('Entrez le symbole du ticker (par ex. AAPL)', 'AAPL')

        start_date = st.date_input('Date de début', value=pd.to_datetime('2022-01-01'))

        end_date = st.date_input('Date de fin', value=pd.to_datetime('today'))

    

        data = yf.download(ticker, start=start_date, end=end_date)


        # Préparer les données pour backtesting.py
        bt_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

        

        with st.spinner('Loading...'):
            if strategy_box == 'RSI, ATR, TRIX & MOMENTUM STRAT':
                # Indicateurs techniques
                def RSI(data, period=14):
                    delta = pd.Series(data).diff(1)
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=period, min_periods=1).mean()
                    avg_loss = loss.rolling(window=period, min_periods=1).mean()
                    rs = avg_gain / avg_loss
                    return 100 - (100 / (1 + rs))

                def ATR(high, low, close, period=14):
                    tr1 = pd.Series(high) - pd.Series(low)
                    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
                    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.rolling(window=period, min_periods=1).mean()
                    return atr

                def Momentum(data, period=12):
                    return pd.Series(data).diff(period)

                def Trix(data, period=9):
                    ema1 = pd.Series(data).ewm(span=period, adjust=False).mean()
                    ema2 = ema1.ewm(span=period, adjust=False).mean()
                    ema3 = ema2.ewm(span=period, adjust=False).mean()
                    return ema3.pct_change() * 100

                # Stratégie de trading
                class RefinedStrategy(Strategy):
                    rsi_period = 14
                    atr_period = 14
                    momentum_period = 12
                    trix_period = 9
                    tolerance = 0

                    def init(self):
                        price = self.data.Close
                        high = self.data.High
                        low = self.data.Low

                        self.rsi = self.I(RSI, price, self.rsi_period)
                        self.atr = self.I(ATR, high, low, price, self.atr_period)
                        self.momentum = self.I(Momentum, price, self.momentum_period)
                        self.trix = self.I(Trix, price, self.trix_period)

                        # Initialiser les plus hautes valeurs
                        self.high_rsi = self.rsi[-1]
                        self.high_atr = self.atr[-1]
                        self.high_momentum = self.momentum[-1]
                        self.high_trix = self.trix[-1]

                    def next(self):
                        # Mettre à jour les plus hautes valeurs précédentes
                        self.high_rsi = max(self.high_rsi, self.rsi[-1])
                        self.high_atr = max(self.high_atr, self.atr[-1])
                        self.high_momentum = max(self.high_momentum, self.momentum[-1])
                        self.high_trix = max(self.high_trix, self.trix[-1])

                        # Calculer le changement des indicateurs avec une marge
                        rsi_change = (self.rsi[-1] - self.rsi[-2]) / self.rsi[-2] if self.rsi[-2] != 0 else 0
                        momentum_change = (self.momentum[-1] - self.momentum[-2]) / self.momentum[-2] if self.momentum[-2] != 0 else 0
                        trix_change = (self.trix[-1] - self.trix[-2]) / self.trix[-2] if self.trix[-2] != 0 else 0

                        # Achat si tous les indicateurs sont positifs après une journée de signal avec tolérance
                        if (rsi_change > self.tolerance and
                            momentum_change > self.tolerance and
                            trix_change < self.tolerance ):
                            self.buy()

                        # Vente si les indicateurs sont inférieurs aux dernières plus hautes valeurs précédentes
                        if (self.rsi[-1] < self.high_rsi - self.tolerance and
                            self.atr[-1] < self.high_atr - self.tolerance and
                            self.momentum[-1] < self.high_momentum - self.tolerance and
                            self.trix[-1] < self.high_momentum - self.tolerance):
                            self.position.close()

                # Exécuter le backtest avec optimisation
                bt = Backtest(bt_data, RefinedStrategy, cash = 10000, commission=.003)
                output = bt.optimize(
                    rsi_period=range(10, 20, 2),
                    atr_period=range(10, 20, 2),
                    momentum_period=range(10, 20, 2),
                    trix_period=range(6, 15, 3),
                    maximize='Equity Final [$]',
                )
                st.write(f"# Backteting pour {ticker}")
                st.write(f"## *RSI, ATR, TRIX & MOMENTUM STRAT* ")
                st.write(f"## Backtesting avec un investissement de 10 000$ CAD")
                st.write(output._strategy)
                st.write(output)
                st.write(f"## *TRADES : RSI, ATR, TRIX & MOMENTUM STRAT* ")
                st.write(output._trades)
                
                if st.button('Télécharger'):
                        plot = custom_plot(bt)
                
            if strategy_box == 'SMA CROSS STRAT':
                import ta
                # Strategy 1
                class SMAcross(Strategy):

                    n1 = 50 #short term window in days
                    n2 = 100 #long term window in days

                    def init(self):
                        close = self.data.Close # Close prices
                        self.sma1 = self.I(ta.trend.sma_indicator, pd.Series(close), self.n1) # short term sma
                        self.sma2 = self.I(ta.trend.sma_indicator, pd.Series(close), self.n2) # long term sma


                    def next(self): # golden cross
                        if crossover(self.sma1, self.sma2):
                            self.buy()
                        elif crossover(self.sma2, self.sma1):
                            self.sell()

                # bt = backtest
                bt1 = Backtest(bt_data, SMAcross, cash=1000, commission = 0.03,
                            exclusive_orders=True)

                output1 = bt1.run()

                optim1 = bt1.optimize(n1 = range(10,160,2), # range of 50 to 160 and cheking every 10 days for short term
                    n2 = range(10,160,2), # range of 50 to 160 and cheking every 10 days for long term
                        maximize = 'Equity Final [$]',
                        max_tries=1000,
                        random_state=0)
                
                st.write(f"# Backteting pour {ticker}")
                st.write(f"## *SMA CROSS STRAT* ")
                st.write(f"## Backtesting avec un investissement de 10 000$ CAD")
                st.write(optim1._strategy)
                st.write(optim1)
                st.write(f"## *TRADES : SMA CROSS STRAT* ")
                st.write(optim1._trades)
                if st.button('Télécharger'):
                    bt1.plot()

            if strategy_box == 'RSI STRAT':
                import pandas_ta as ta
                import pandas as pd


                class RsiOscillator(Strategy):

                    upper_bound = 70
                    lower_bound = 30
                    rsi_window = 14

                    # Do as much initial computation as possible
                    def init(self):
                        # Use pandas_ta to calculate RSI
                        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), self.rsi_window)

                    # Step through bars one by one
                    # Note that multiple buys are a thing here
                    def next(self):
                        if crossover(self.rsi, self.upper_bound):
                            self.position.close()
                        elif crossover(self.lower_bound, self.rsi):
                            self.buy()

                bt2 = Backtest(bt_data, RsiOscillator, cash=1000, commission=.002)
                stats3 = bt2.optimize(
                        upper_bound = range(50,85,2),
                        lower_bound = range(15,45,2),
                        rsi_window = range(10,30,2),
                        maximize='Equity Final [$]')
                stats3 = bt2.run()

                st.write(f"# Backteting pour {ticker}")
                st.write(f"## *RSI STRAT* ")
                st.write(f"## Backtesting avec un investissement de 10 000$ CAD")
                st.write(f"## Stratégie")
                st.write(stats3._strategy)
                st.write(f"## Taux de réussite")
                st.write(stats3['Win Rate [%]'])
                st.write(f"## Retours sur investissement")
                st.write(stats3['Return [%]'])
                st.write('## *Informations*')
                st.write(stats3)
                st.write(f"## *TRADES : RSI STRAT* ")
                st.write(stats3._trades)

                if st.button('Télécharger'):
                    bt2.plot()

    if app_mode == "Backtesting - StockGenius":

        import streamlit as st
        import numpy as np
        import pandas as pd
        import yfinance as yf
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        import tensorflow as tf
        from tensorflow.keras import layers, models
        import plotly.graph_objects as go

        
        # Function to plot the price with trade markers (buy and sell)
        def plot_trades_on_price(data, trades, ticker):
            fig = go.Figure()

            # Plot the closing prices
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))

            # Plot buy trades as green markers
            buy_trades = trades[trades['Size'] > 0]
            fig.add_trace(go.Scatter(
                x=buy_trades['ExitTime'], 
                y=buy_trades['ExitPrice'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Trade (Achat/Vente)'
            ))

            # Plot sell trades as red markers
            sell_trades = trades[trades['Size'] < 0]
            fig.add_trace(go.Scatter(
                x=sell_trades['ExitTime'], 
                y=sell_trades['ExitPrice'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Short (Vente/Achat)'
            ))

            # Update the layout
            fig.update_layout(
                title=f"Price and Trades for {ticker}",
                xaxis_title="Date",
                yaxis_title="Price",
                legend=dict(x=0, y=1, traceorder='normal'),
                hovermode="x unified",
            )

            # Add annotation for branding
            fig.add_annotation(
                text='StockGenius',
                xref='paper', yref='paper',
                x=0.01, y=0.01,
                showarrow=False,
                font=dict(size=12, color='white'),
                align='right'
            )

            # Display the chart
            st.plotly_chart(fig)

        # Running the backtest and plotting both the equity curve and trades
        def run_optimized_backtest(ticker, start_date, end_date, parameter_grid, cash=10_000):
            set_seed(42)

            # Download historical data
            data = download_stock_data(ticker, start_date, end_date)
            st.write("Données historiques :")
            st.write(data.head())  # Show the first few rows of data

            # Optimization
            best_params, best_equity = optimize_strategy(data, parameter_grid)
            #print(f"Meilleurs paramètres : {best_params}")
            #print(f"Equity Final (optimisation) : {best_equity}")

            # Apply the best parameters to the strategy
            for key, value in best_params.items():
                setattr(KANStrategy, key, value)

            # Run the backtest with the best parameters
            bt = Backtest(data, KANStrategy, cash=cash, commission=0.001)
            results = bt.run()

            # Display backtest results
            st.write("Résultat du backtest :")
            st.write(results)

            final_equity_backtest = results['Equity Final [$]']
            #st.write(f"Equity Final (backtest final) : {final_equity_backtest}")

            
            # Plot the equity curve
            plot_equity_curve(results['_equity_curve'], ticker)

            # Display and plot trades
            trades = results._trades
            st.write("Détails des Trades Exécutés :")
            st.dataframe(trades)

            # Plot trades on price chart
            plot_trades_on_price(data, trades, ticker)

        # Fonction pour télécharger les données historiques des actions
        def download_stock_data(ticker, start_date, end_date):
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            return data['Close']

        # Kolmogorov-Arnold Network (KAN) Model
        def build_kan_model(input_dim):
            model = models.Sequential()
            model.add(layers.Input(shape=(input_dim,)))
            model.add(layers.Dense(128, activation='tanh'))
            model.add(layers.Dense(64, activation='tanh'))
            model.add(layers.Dense(32, activation='tanh'))
            model.add(layers.Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model

        # Préparer les données pour l'entraînement
        def prepare_data(data, window_size=252):
            X = []
            y = []
            for i in range(window_size, len(data)):
                X.append(data[i-window_size:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        # Fonction pour tracer le graphique des prédictions
        def plot_predictionsKAN(real_prices, predictions):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(real_prices))),
                                    y=real_prices.flatten(), mode='lines', name='Prix réel'))
            fig.add_trace(go.Scatter(x=list(range(len(predictions))),
                                    y=predictions.flatten(), mode='lines', name='Prix prédit'))
            fig.update_layout(title='Prédiction des prix avec Deep Learning', 
                            xaxis_title='Temps', yaxis_title='Prix', hovermode='x unified')
            
            fig.add_annotation(
            text='StockGenius',
            xref='paper', yref='paper',
            x=0.01, y=0.01,
            showarrow=False,
            font=dict(size=12, color='white'),
            align='left'
        )
            

            st.plotly_chart(fig)
            st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")

        # Fonction pour calculer le taux de réussite (win rate)
        def calculate_win_rate(real_prices, predicted_prices):
            correct_predictions = np.sum((predicted_prices[1:] > predicted_prices[:-1]) == (real_prices[1:] > real_prices[:-1]))
            total_predictions = len(real_prices) - 1
            win_rate = correct_predictions / total_predictions
            return win_rate



        # Fonction pour tracer la prédiction future avec un graphique de chandeliers
        def plot_predictionKAN(ticker, forecast_days, predicted_price, win_rate):
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            hist['30D_MA'] = hist['Close'].rolling(window=30).mean()
            future_date = hist.index[-1] + pd.Timedelta(days=forecast_days)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Prix Historique'))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['30D_MA'], mode='lines', line=dict(color='blue', width=2), name='Moyenne Mobile 30 Jours'))
            fig.add_trace(go.Scatter(x=[future_date], y=predicted_price, mode='markers', marker=dict(color='red', size=10), name=f'Prix Prédit dans {forecast_days} Jours'))
            fig.add_annotation(x=future_date, y=float(predicted_price[0]), text=f'Prix Prédit: ${float(predicted_price[0]):.2f}\nTaux de Réussite: {win_rate:.2%}', showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(size=12, color='red'), align='center')

            fig.update_layout(title=f'Prédiction des Prix de {ticker} pour {forecast_days} Jours', xaxis_title='Date', yaxis_title='Prix', xaxis_rangeslider_visible=False, plot_bgcolor='white', hovermode='x unified')

            fig.add_annotation(
            text='StockGenius',
            xref='paper', yref='paper',
            x=0.01, y=0.01,
            showarrow=False,
            font=dict(size=12, color='black'),
            align='left'
        )
            
            st.plotly_chart(fig)
            st.write(f"*Avertissement : Ce graphique est fourni à titre informatif seulement et ne doit pas être utilisé pour prendre des décisions financières. Utilisation à des fins personnelles uniquement.")

        st.title(f":bar_chart: Modèle Deep Learning StockGenius")

        # Sélection de l'utilisateur
        ticker = st.text_input("Entrez le symbole de l'action (ex: AAPL):", value='AAPL')
        start_date = pd.to_datetime('2010-01-01')
        end_date = dt.date.today()
        forecast_days = 7
        window_size = 60

        # Télécharger les données historiques
        data = download_stock_data(ticker, start_date, end_date)

        # Normalisation des données
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

        # Préparer les données d'entraînement
        X, y = prepare_data(scaled_data, window_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Créer et entraîner le modèle
        model = build_kan_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=32)

        # Prédiction sur les données de test
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calcul du taux de réussite (win rate)
        win_rate = calculate_win_rate(real_prices, predictions)

        # Sauvegarder les prédictions dans un fichier CSV
        save_to_csv({
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'predicted_price': predictions[-1][0],
            'win_rate': win_rate
        }, folder='Predictions', filename='predictions.csv')

        # Affichage des résultats
        st.subheader("Résultats des Prédictions")
        plot_predictionsKAN(real_prices, predictions)

        # Prédiction des prix futurs
        last_window = scaled_data[-window_size:]
        last_window = np.expand_dims(last_window, axis=0)
        predicted_future_price = model.predict(last_window)
        predicted_future_price = scaler.inverse_transform(predicted_future_price)

        # Affichage de la prédiction future
        
        plot_predictionKAN(ticker, forecast_days, [predicted_future_price], win_rate)

        import numpy as np
        import random
        import pandas as pd
        import yfinance as yf
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        import os
        from backtesting import Backtest, Strategy
        import streamlit as st
        import plotly.graph_objects as go


        # Fixer les graines pour assurer la reproductibilité
        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)

        # Fonction pour télécharger les données historiques des actions
        def download_stock_data(ticker, start_date, end_date):
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            return data

        # Kolmogorov-Arnold Network (KAN) Model
        def build_kan_model(input_dim, neurons_layer1, neurons_layer2, neurons_layer3, optimizer, loss_function):
            model = models.Sequential()
            model.add(layers.Input(shape=(input_dim,)))
            model.add(layers.Dense(neurons_layer1, activation='tanh'))
            model.add(layers.Dense(neurons_layer2, activation='tanh'))
            model.add(layers.Dense(neurons_layer3, activation='tanh'))  # Troisième couche
            model.add(layers.Dense(1))  # Sortie avec une unité
            model.compile(optimizer=optimizer, loss=loss_function)
            return model

        # Préparer les données pour l'entraînement
        def prepare_data(data, window_size=252):
            X = []
            y = []
            for i in range(window_size, len(data)):
                X.append(data[i-window_size:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        # Stratégie de backtesting avec modèle KAN
        class KANStrategy(Strategy):
            window_size = 60
            neurons_layer1 = 128
            neurons_layer2 = 64
            neurons_layer3 = 32  # Nouvelle couche
            n_training_runs = 5
            stop_loss_pct = 0.02
            take_profit_pct = 0.04
            slippage_pct = 0.001
            optimizer = 'adam'
            loss_function = 'mean_squared_error'
            epochs = 50
            batch_size = 32

            def init(self):
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                close_prices = np.array(self.data.Close)
                scaled_data = self.scaler.fit_transform(close_prices.reshape(-1, 1))

                # Préparer les données pour le modèle KAN
                X, y = prepare_data(scaled_data, self.window_size)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

                # Entraîner plusieurs modèles pour le bagging
                self.models = []
                for _ in range(self.n_training_runs):
                    model = build_kan_model(X_train.shape[1], self.neurons_layer1, self.neurons_layer2, self.neurons_layer3, self.optimizer, self.loss_function)
                    model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
                    self.models.append(model)

            def next(self):
                if len(self.data.Close) > self.window_size:
                    data = np.array(self.data.Close[-self.window_size:])
                    scaled_data = self.scaler.transform(data.reshape(-1, 1)).reshape(1, -1)

                    # Moyenne des prédictions de tous les modèles entraînés
                    predictions = np.mean([model.predict(scaled_data) for model in self.models], axis=0)
                    predicted_price = self.scaler.inverse_transform(predictions)[0, 0]
                    current_price = self.data.Close[-1]

                    # Logique pour acheter/vendre basé sur la prédiction moyenne
                    if predicted_price > current_price:
                        buy_price = current_price * (1 + self.slippage_pct)
                        if not self.position:
                            self.buy(sl=buy_price * (1 - self.stop_loss_pct), tp=buy_price * (1 + self.take_profit_pct))
                        elif self.position.is_short:
                            self.position.close()
                            self.buy(sl=buy_price * (1 - self.stop_loss_pct), tp=buy_price * (1 + self.take_profit_pct))

                    elif predicted_price < current_price:
                        sell_price = current_price * (1 - self.slippage_pct)
                        if self.position.is_long:
                            self.position.close()
                        elif not self.position:
                            self.sell(sl=sell_price * (1 + self.stop_loss_pct), tp=sell_price * (1 - self.take_profit_pct))

        # Fonction d'optimisation pour trouver les meilleurs paramètres
        def optimize_strategy(data, parameter_grid):
            best_equity = -np.inf
            best_params = None

            for params in parameter_grid:
                # Mettre à jour les hyperparamètres dans la stratégie
                KANStrategy.window_size = params['window_size']
                KANStrategy.neurons_layer1 = params['neurons_layer1']
                KANStrategy.neurons_layer2 = params['neurons_layer2']
                KANStrategy.neurons_layer3 = params['neurons_layer3']  # Troisième couche ajoutée
                KANStrategy.optimizer = params['optimizer']
                KANStrategy.loss_function = params['loss_function']
                KANStrategy.epochs = params['epochs']
                KANStrategy.batch_size = params['batch_size']
                KANStrategy.stop_loss_pct = params['stop_loss_pct']
                KANStrategy.take_profit_pct = params['take_profit_pct']
                KANStrategy.slippage_pct = params['slippage_pct']

                # Lancer le backtest avec ces paramètres
                bt = Backtest(data, KANStrategy, cash=10_000, commission=0.001)
                result = bt.run()  # Ici on définit 'result'

                

                # Vérifier si l'Equity Final est supérieur au meilleur equity actuel
                if result['Equity Final [$]'] > best_equity:
                    best_equity = result['Equity Final [$]']
                    best_params = params

            return best_params, best_equity

        # Fonction pour tracer l'équité uniquement
        def plot_equity_curve(equity_curve, ticker):
            # Tracer la courbe d'équité
            fig = go.Figure()
            
            # Tracer l'équité
            fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve['Equity'], mode='lines', name='Equity Curve', line=dict(color='blue')))
            
            # Mettre à jour la mise en page
            fig.update_layout(
                title=f"Rendement du capital pour {ticker}",
                xaxis_title="Date",
                yaxis_title="Capital",
                legend=dict(x=0, y=1, traceorder='normal'),
                hovermode="x unified",
            )

            fig.add_annotation(
            text='StockGenius',
            xref='paper', yref='paper',
            x=0.01, y=0.01,
            showarrow=False,
            font=dict(size=12, color='white'),
            align='right'
        )
            
            # Afficher le graphique
            st.plotly_chart(fig)

        # Lancer le backtest et tracer uniquement la courbe d'équité
        def run_optimized_backtest(ticker, start_date, end_date, parameter_grid, cash=10_000):
            set_seed(42)

            # Télécharger les données historiques
            data = download_stock_data(ticker, start_date, end_date)
            
            # Optimisation
            best_params, best_equity = optimize_strategy(data, parameter_grid)
            

            # Appliquer les meilleurs paramètres à la stratégie
            for key, value in best_params.items():
                setattr(KANStrategy, key, value)

            # Lancer le backtest avec les meilleurs paramètres
            bt = Backtest(data, KANStrategy, cash=cash, commission=0.001)
            results = bt.run()

            
            final_equity_backtest = results['Equity Final [$]']
            

            

            # Tracer la courbe d'équité
            plot_equity_curve(results['_equity_curve'], ticker)

            # Afficher les trades sous forme de tableau
            trades = results._trades
            # Après avoir calculé les prédictions et exécuté le backtest
            plot_trades_on_price(data, trades, ticker)


            st.write("Détails des Trades Exécutés :")
            st.dataframe(results)
            st.dataframe(trades)


            print(f"Meilleurs paramètres : {best_params}")
            


        
        # Streamlit UI
        st.title(f":chart_with_upwards_trend: Backtesting Modèle Deep Learning StockGenius")

        ticker = st.text_input("Entrez le ticker de l'action (ex: AAPL)", value="AAPL")
        start_date = st.date_input("Date de début", pd.to_datetime("2024-05-01"))
        end_date
        cash = st.number_input("Montant initial", min_value=0.0, value=10_000.0, step=100.0)


        # Paramètres de grille pour optimisation
        parameter_grid = [
            {'window_size': 50, 'neurons_layer1': 512, 'neurons_layer2': 256, 'neurons_layer3': 128, 'optimizer': 'adam', 'loss_function': 'mean_squared_error', 'epochs': 100, 'batch_size': 32, 'stop_loss_pct': 0.01, 'take_profit_pct': 0.05, 'slippage_pct': 0.001},
            {'window_size': 60, 'neurons_layer1': 512, 'neurons_layer2': 256, 'neurons_layer3': 128, 'optimizer': 'adam', 'loss_function': 'mean_squared_error', 'epochs': 150, 'batch_size': 64, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.06, 'slippage_pct': 0.002},
            {'window_size': 70, 'neurons_layer1': 256, 'neurons_layer2': 128, 'neurons_layer3': 64, 'optimizer': 'rmsprop', 'loss_function': 'huber', 'epochs': 150, 'batch_size': 64, 'stop_loss_pct': 0.015, 'take_profit_pct': 0.07, 'slippage_pct': 0.003},
            {'window_size': 80, 'neurons_layer1': 128, 'neurons_layer2': 64, 'neurons_layer3': 32, 'optimizer': 'adam', 'loss_function': 'mean_absolute_error', 'epochs': 200, 'batch_size': 128, 'stop_loss_pct': 0.025, 'take_profit_pct': 0.08, 'slippage_pct': 0.004},
            {'window_size': 90, 'neurons_layer1': 512, 'neurons_layer2': 256, 'neurons_layer3': 128, 'optimizer': 'sgd', 'loss_function': 'mean_squared_error', 'epochs': 100, 'batch_size': 32, 'stop_loss_pct': 0.01, 'take_profit_pct': 0.05, 'slippage_pct': 0.001},
            {'window_size': 100, 'neurons_layer1': 256, 'neurons_layer2': 128, 'neurons_layer3': 64, 'optimizer': 'adam', 'loss_function': 'mean_absolute_error', 'epochs': 150, 'batch_size': 64, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.06, 'slippage_pct': 0.002}
        ]

        if st.button("Lancer le backtest"):
            with st.spinner('Loading...'):
                run_optimized_backtest(ticker, start_date, end_date, parameter_grid)
                       
    if app_mode == "Macro":
        import streamlit as st
        import yfinance as yf
        import pandas as pd
        import plotly.graph_objects as go
        from datetime import date

        # Titre de l'application avec un style soigné
        st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>Indices économiques sectoriels</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Suivi des secteurs financiers - USA, Canada, Chine, Europe, Inde</h4>", unsafe_allow_html=True)

        # Sidebar pour sélectionner un pays, un secteur et une plage de dates
        st.sidebar.markdown("<h3 style='color: #FFFFFF;'>Sélection des paramètres</h3>", unsafe_allow_html=True)

        # Pays disponibles
        countries = ['USA', 'Canada', 'Chine', 'Europe', 'Inde']
        selected_country = st.sidebar.selectbox('Sélectionner un pays', countries)

        # Map pour obtenir les tickers des indices sectoriels via Yahoo Finance
        sectors = {
            'USA': {
                'Indice global (S&P 500)': '^GSPC',
                'Secteur financier': '^SP500-40',
                'Secteur des énergies': '^SP500-10',
                'Secteur des technologies': '^SP500-45',
            },
            'Canada': {
                'Indice global (S&P/TSX)': '^GSPTSE',
                'Secteur financier': '^TSXFN',
                'Secteur des énergies': '^TSXEN',
                'Secteur des matériaux': '^TSXMT',
            },
            'Chine': {
                'Indice global (Shanghai Composite)': '000001.SS',
                'Secteur financier': 'CNAFIN',
                'Secteur des énergies': 'CNAENE',
                'Secteur des matériaux': 'CNAMAT',
            },
            'Europe': {
                'Indice global (Euro Stoxx 50)': '^STOXX50E',
                'Secteur financier': '^STOXX50F',
                'Secteur des énergies': '^STOXX50E-10',
            },
            'Inde': {
                'Indice global (Nifty 50)': '^NSEI',
                'Secteur financier': 'CNXFIN',
                'Secteur des énergies': 'CNXENE',
            }
        }

        # Sélection du secteur
        selected_sector = st.sidebar.selectbox('Sélectionner un secteur', list(sectors[selected_country].keys()))

        # Sélection de la plage de dates
        start_date = st.sidebar.date_input('Date de début', value=date(2022, 1, 1))
        end_date = st.sidebar.date_input('Date de fin', value=date.today())

        # Bouton pour lancer la recherche des données
        if st.sidebar.button("Charger les données"):
            
            # Ticker du secteur
            ticker = sectors[selected_country][selected_sector]

            # Débogage
            with st.expander("Informations"):
                st.write(f"Pays sélectionné : {selected_country}")
                st.write(f"Secteur sélectionné : {selected_sector}")
                st.write(f"Ticker utilisé : {ticker}")
                st.write(f"Plage de dates : {start_date} au {end_date}")
            
            # Télécharger les données historiques via yfinance
            try:
                df = yf.download(ticker, start=start_date, end=end_date)

                # Vérification si des données sont disponibles
                if df.empty:
                    st.error(f"Aucune donnée disponible pour {selected_sector} sur la période sélectionnée.")
                else:
                    # Afficher les données en tableau avec un style plus sobre
                    st.markdown(f"<h3 style='color: #FFFFFF;'>Données historiques pour {selected_sector}</h3>", unsafe_allow_html=True)
                    st.dataframe(df.style.set_properties(**{'background-color': '#FAFAFA', 'color': '#000000'}))
                    
                    # Graphique en chandeliers
                    fig = go.Figure(data=[go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Prix'
                    )])

                    # Personnalisation du graphique
                    fig.update_layout(
                        title=f'Graphique en chandeliers pour {selected_sector}',
                        xaxis_title='Date',
                        yaxis_title='Prix',
                        xaxis_rangeslider_visible=False,
                        template='plotly_dark'
                    )

                    # Affichage du graphique avec Plotly
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur lors de la récupération des données pour {selected_sector} : {e}")
                st.write(f"Message d'erreur complet : {e}")
        else:
            st.info("Sélectionnez les paramètres dans la barre latérale et cliquez sur 'Charger les données'.")

    
    if app_mode == "Test":
        import streamlit as st
        import pandas as pd
        import math
        from pathlib import Path

        

        # -----------------------------------------------------------------------------
        # Declare some useful functions.

        @st.cache_data
        def get_gdp_data():
            """Grab GDP data from a CSV file.

            This uses caching to avoid having to read the file every time. If we were
            reading from an HTTP endpoint instead of a file, it's a good idea to set
            a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
            """

            # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
            DATA_FILENAME = Path(__file__).parent/'gdp_data.csv'
            raw_gdp_df = pd.read_csv(DATA_FILENAME)

            MIN_YEAR = 1960
            MAX_YEAR = 2022

            # The data above has columns like:
            # - Country Name
            # - Country Code
            # - [Stuff I don't care about]
            # - GDP for 1960
            # - GDP for 1961
            # - GDP for 1962
            # - ...
            # - GDP for 2022
            #
            # ...but I want this instead:
            # - Country Name
            # - Country Code
            # - Year
            # - GDP
            #
            # So let's pivot all those year-columns into two: Year and GDP
            gdp_df = raw_gdp_df.melt(
                ['Country Code'],
                [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
                'Year',
                'GDP',
            )

            # Convert years from string to integers
            gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

            return gdp_df

        gdp_df = get_gdp_data()

        # -----------------------------------------------------------------------------
        # Draw the actual page

        # Set the title that appears at the top of the page.
        '''
        # :earth_americas: GDP dashboard

        Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
        notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
        But it's otherwise a great (and did I mention _free_?) source of data.
        '''

        # Add some spacing
        ''
        ''

        min_value = gdp_df['Year'].min()
        max_value = gdp_df['Year'].max()

        from_year, to_year = st.slider(
            'Which years are you interested in?',
            min_value=min_value,
            max_value=max_value,
            value=[min_value, max_value])

        countries = gdp_df['Country Code'].unique()

        if not len(countries):
            st.warning("Select at least one country")

        selected_countries = st.multiselect(
            'Which countries would you like to view?',
            countries,
            ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

        ''
        ''
        ''

        # Filter the data
        filtered_gdp_df = gdp_df[
            (gdp_df['Country Code'].isin(selected_countries))
            & (gdp_df['Year'] <= to_year)
            & (from_year <= gdp_df['Year'])
        ]

        st.header('GDP over time', divider='gray')

        ''

        st.line_chart(
            filtered_gdp_df,
            x='Year',
            y='GDP',
            color='Country Code',
        )

        ''
        ''


        first_year = gdp_df[gdp_df['Year'] == from_year]
        last_year = gdp_df[gdp_df['Year'] == to_year]

        st.header(f'GDP in {to_year}', divider='gray')

        ''

        cols = st.columns(4)

        for i, country in enumerate(selected_countries):
            col = cols[i % len(cols)]

            with col:
                first_gdp = first_year[gdp_df['Country Code'] == country]['GDP'].iat[0] / 1000000000
                last_gdp = last_year[gdp_df['Country Code'] == country]['GDP'].iat[0] / 1000000000

                if math.isnan(first_gdp):
                    growth = 'n/a'
                    delta_color = 'off'
                else:
                    growth = f'{last_gdp / first_gdp:,.2f}x'
                    delta_color = 'normal'

                st.metric(
                    label=f'{country} GDP',
                    value=f'{last_gdp:,.0f}B',
                    delta=growth,
                    delta_color=delta_color
                )
    
    if app_mode == "Contact":

        import streamlit as st
        import yagmail

        # Fonction pour envoyer un email via yagmail
        def send_email(name, email, message):
            receiver = "antoinefebresgagne@gmail.com"  # Remplacez par votre adresse email
            subject = f"Message de {name} via l'application Streamlit"
            body = f"Nom: {name}\nEmail: {email}\n\nMessage:\n{message}"

            try:
                # Connexion avec yagmail, ici vous devrez avoir configuré yagmail avec vos identifiants
                yag = yagmail.SMTP("votre_adresse_email@example.com", "votre_mot_de_passe")
                yag.send(to=receiver, subject=subject, contents=body)
                return True
            except Exception as e:
                st.error(f"Erreur lors de l'envoi de l'email : {e}")
                return False

        # Page de contact
        st.title(f"Contactez-moi! :mailbox_with_mail:")

        # Formulaire de contact
        with st.form("contact_form"):
            name = st.text_input("Nom")
            email = st.text_input("Email")
            message = st.text_area("Votre message")

            submitted = st.form_submit_button("Envoyer")

            if submitted:
                if name and email and message:
                    # Envoyer l'email
                    success = send_email(name, email, message)
                    if success:
                        st.success("Votre message a été envoyé avec succès !")
                    else:
                        st.error("Une erreur est survenue lors de l'envoi de votre message. Veuillez réessayer.")
                else:
                    st.error("Veuillez remplir tous les champs avant d'envoyer.")