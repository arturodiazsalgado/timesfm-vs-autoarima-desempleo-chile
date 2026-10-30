"""Funciones del análisis de pronóstico del desempleo."""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string


def descargar_mindicador(indicador='tasa_desempleo', desde=2009, hasta=None,
                         col_valor='desempleo', pausa=0.2):
    """Descarga una serie mensual desde mindicador.cl."""
    import requests

    hasta = hasta or pd.Timestamp.today().year
    filas = []
    for anio in range(desde, hasta + 1):
        r = requests.get(f'https://mindicador.cl/api/{indicador}/{anio}', timeout=90)
        if r.status_code == 200:
            filas += [{'fecha': x['fecha'], col_valor: x['valor']} for x in r.json().get('serie', [])]
        time.sleep(pausa)

    df = pd.DataFrame(filas)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df[col_valor] = pd.to_numeric(df[col_valor], errors='coerce')
    return df.sort_values('fecha').drop_duplicates('fecha').reset_index(drop=True)


def preparar_serie(df, col_fecha='fecha', col_valor='desempleo', inicio=None, fin=None):
    """Normaliza las fechas a inicio de mes y verifica continuidad."""
    fecha = pd.to_datetime(df[col_fecha])
    if fecha.dt.tz is not None:
        fecha = fecha.dt.tz_convert('America/Santiago').dt.tz_localize(None)
    fecha = fecha.dt.to_period('M').dt.to_timestamp()

    serie = (pd.Series(df[col_valor].to_numpy(dtype=float),
                       index=pd.DatetimeIndex(fecha), name=col_valor)
               .sort_index()
               .asfreq('MS')
               .loc[inicio:fin])

    faltantes = serie[serie.isna()].index
    if len(faltantes):
        raise ValueError(f"{len(faltantes)} meses faltantes, ej.: "
                         f"{faltantes[:5].strftime('%Y-%m').tolist()}")
    return serie

REGIONES_15 = {
    1: 'Tarapacá', 2: 'Antofagasta', 3: 'Atacama', 4: 'Coquimbo', 5: 'Valparaíso',
    6: "O'Higgins", 7: 'Maule', 8: 'Biobío', 9: 'Araucanía', 10: 'Los Lagos',
    11: 'Aysén', 12: 'Magallanes', 13: 'Metropolitana', 14: 'Los Ríos',
    15: 'Arica y Parinacota', 99: 'Total país',
}


def cargar_series_regionales(ruta, regiones=(5, 99), col_region='region_15',
                             col_valor='tasa_desempleo', en_porcentaje=True, inicio=None, fin=None):
    """Carga series mensuales por región de la ENE."""
    df = pd.read_csv(ruta, parse_dates=['fecha'])
    series = {}
    for r in regiones:
        s = preparar_serie(df[df[col_region] == r], col_valor=col_valor, inicio=inicio, fin=fin)
        if en_porcentaje:
            s = s * 100
        s.name = REGIONES_15.get(r, str(r))
        series[s.name] = s
    return series


def comparar_series(a, b, rezagos=range(-3, 4)):
    """Compara dos series bajo distintos rezagos mensuales."""
    filas = []
    for k in rezagos:
        d = pd.concat([a, b.shift(k)], axis=1, keys=['a', 'b']).dropna()
        filas.append({'rezago': k, 'n': len(d),
                      'dif_media_abs': (d['a'] - d['b']).abs().mean(),
                      'dif_max_abs': (d['a'] - d['b']).abs().max(),
                      'corr_variaciones': d['a'].diff().corr(d['b'].diff())})
    return pd.DataFrame(filas).set_index('rezago')

VENTANA_PANDEMIA = ('2020-03-01', '2021-09-01')


def pulsos(index, inicio, fin):
    """Genera un indicador de pulso para cada mes del intervalo."""
    meses = pd.date_range(inicio, fin, freq='MS')
    return pd.DataFrame({f'p_{m:%Y_%m}': (index == m).astype(float) for m in meses}, index=index)


def configurar_autoarima(**kwargs):
    """Configura AutoARIMA con estacionalidad anual."""
    params = dict(season_length=12, d=None, D=None,
                  start_p=0, start_q=0, max_p=3, max_q=3,
                  start_P=0, start_Q=0, max_P=2, max_Q=2,
                  ic='aic', stepwise=True)
    params.update(kwargs)
    return AutoARIMA(**params)


def _exogenas(train_index, h, ventana_pulsos):
    X = pulsos(train_index, *ventana_pulsos)
    X = X.loc[:, X.sum() > 0]
    if X.shape[1] == 0:
        return None, None
    return X.to_numpy(), np.zeros((h, X.shape[1]))


def pronostico_un_origen(serie, origen, h=12, ventana=None, ventana_pulsos=None,
                         nombre=None, **kwargs_arima):
    """Estima ARIMA(X) hasta el origen y pronostica h meses."""
    origen = pd.Timestamp(origen)
    train = serie.loc[:origen]
    if ventana is not None:
        train = train.iloc[-ventana:]

    X_train, X_futuro = (None, None)
    if ventana_pulsos is not None:
        X_train, X_futuro = _exogenas(train.index, h, ventana_pulsos)

    modelo = configurar_autoarima(**kwargs_arima).fit(y=train.to_numpy(), X=X_train)
    fechas = pd.date_range(origen + pd.DateOffset(months=1), periods=h, freq='MS')

    return pd.DataFrame({
        'origen': origen,
        'fecha': fechas,
        'horizonte': np.arange(1, h + 1),
        'pronostico': modelo.predict(h=h, X=X_futuro)['mean'],
        'observado': serie.reindex(fechas).to_numpy(),
        'modelo': nombre or ('AutoARIMAX' if ventana_pulsos is not None else 'AutoARIMA'),
        'especificacion': arima_string(modelo.model_).strip(),
        'n_obs': len(train),
        'n_pulsos': 0 if X_train is None else X_train.shape[1],
    })


def pronostico_rolling(serie, ultimo_origen, primer_origen=None, h=12, ventana=None,
                       ventana_pulsos=None, nombre=None, min_obs=36, n_jobs=-1,
                       verbose=True, **kwargs_arima):
    """Reestima ARIMA(X) para cada origen mensual."""
    if primer_origen is None:
        primer_origen = serie.index[0] + pd.DateOffset(months=min_obs - 1)
    origenes = pd.date_range(primer_origen, ultimo_origen, freq='MS')

    if origenes[0] < serie.index[0] + pd.DateOffset(months=min_obs - 1):
        raise ValueError(f"El primer origen ({origenes[0]:%Y-%m}) deja menos de {min_obs} obs. de estimación")
    if origenes[-1] > serie.index[-1]:
        raise ValueError(f"El último origen ({origenes[-1]:%Y-%m}) es posterior al último dato")

    t0 = time.time()
    resultados = Parallel(n_jobs=n_jobs)(
        delayed(pronostico_un_origen)(serie, origen, h=h, ventana=ventana,
                                      ventana_pulsos=ventana_pulsos, nombre=nombre, **kwargs_arima)
        for origen in origenes[::-1]
    )
    res = (pd.concat(resultados, ignore_index=True)
             .sort_values(['origen', 'horizonte'], ignore_index=True))
    if verbose:
        print(f"{res['modelo'].iloc[0]}: {len(origenes)} modelos estimados en {time.time() - t0:.0f} s")
    return res


def ajustar_y_resumir(serie, ventana_pulsos=None, **kwargs_arima):
    """Reestima la especificación elegida en statsmodels."""
    from statsmodels.tsa.arima.model import ARIMA

    X = None
    if ventana_pulsos is not None:
        X = pulsos(serie.index, *ventana_pulsos)
        X = X.loc[:, X.sum() > 0]

    modelo = configurar_autoarima(**kwargs_arima).fit(
        y=serie.to_numpy(), X=None if X is None else X.to_numpy())
    p, q, P, Q, m, d, D = modelo.model_['arma']
    coefs = modelo.model_['coef']
    trend = 't' if 'drift' in coefs else ('c' if 'intercept' in coefs else 'n')

    res_sm = ARIMA(serie, exog=X, order=(p, d, q), seasonal_order=(P, D, Q, m), trend=trend).fit()
    print(arima_string(modelo.model_).strip())
    return modelo, res_sm


def pronostico_rolling_naive(serie, res_referencia, estacional=False):
    """Pronostica con naive simple o estacional usando los mismos orígenes."""
    r = res_referencia[['origen', 'fecha', 'horizonte', 'observado']].copy()
    if estacional:
        rezagos = 12 * np.ceil(r['horizonte'] / 12).astype(int)
        fechas_base = [f - pd.DateOffset(months=int(k)) for f, k in zip(r['fecha'], rezagos)]
    else:
        fechas_base = r['origen']
    r['pronostico'] = serie.reindex(fechas_base).to_numpy()
    if r['pronostico'].isna().any():
        raise ValueError('Faltan observaciones para construir el pronóstico naive')
    r['modelo'] = 'Naive estacional' if estacional else 'Naive'
    return r

REPO_25 = 'google/timesfm-2.5-200m-pytorch'
REPO_3 = 'google/timesfm-3.0-pytorch'


def cargar_timesfm25(max_context=256, per_core_batch_size=32):
    """Carga TimesFM 2.5 en PyTorch."""
    import timesfm

    modelo = timesfm.TimesFM_2p5_200M_torch.from_pretrained(REPO_25, torch_compile=False)
    modelo.compile(timesfm.ForecastConfig(
        max_context=max_context,
        max_horizon=128,
        normalize_inputs=True,
        per_core_batch_size=per_core_batch_size,
        force_flip_invariance=True,
        infer_is_positive=True,
    ))
    return modelo


def cargar_timesfm3(device='cpu', per_core_batch_size=16):
    """Carga TimesFM 3.0."""
    from timesfm3 import TimesFM3Forecaster

    return TimesFM3Forecaster.from_pretrained(REPO_3, device=device,
                                              per_core_batch_size=per_core_batch_size)


def _es_timesfm3(modelo):
    return type(modelo).__name__ == 'TimesFM3Forecaster'


def nombre_timesfm(modelo):
    return 'TimesFM 3.0' if _es_timesfm3(modelo) else 'TimesFM 2.5'


def _pulsos_timesfm(indices, h, ventana_pulsos):
    """Covariables observadas hasta cada origen; futuro fijado en cero."""
    nombres = [f'p_{m:%Y_%m}' for m in pd.date_range(*ventana_pulsos, freq='MS')]
    matrices = []
    for idx in indices:
        x = pulsos(idx, *ventana_pulsos).reindex(columns=nombres).to_numpy(dtype=np.float32).T
        matrices.append(np.pad(x, ((0, 0), (0, h)), mode='constant'))
    return nombres, matrices


def habilitar_xreg_timesfm25(modelo):
    """XReg de TimesFM 2.5 requiere backcast en la configuración."""
    from dataclasses import replace

    cfg = modelo.forecast_config
    if cfg is None:
        raise ValueError('Primero hay que compilar TimesFM 2.5')
    if not cfg.return_backcast:
        modelo.compile(replace(cfg, return_backcast=True))
    return modelo


def predecir_timesfm(modelo, contextos, h, indices=None, ventana_pulsos=None):
    """Pronóstico TimesFM univariado o con pulsos de intervención."""
    contextos = [np.asarray(c, dtype=np.float32) for c in contextos]
    if ventana_pulsos is not None:
        if indices is None or len(indices) != len(contextos):
            raise ValueError('Las fechas de los contextos son necesarias para usar covariables')
        nombres, matrices = _pulsos_timesfm(indices, h, ventana_pulsos)
        if any(m.shape[1] != len(c) + h for m, c in zip(matrices, contextos)):
            raise ValueError('Las covariables no están alineadas con la serie')

    if _es_timesfm3(modelo):
        opciones = {}
        if ventana_pulsos is not None:
            opciones['past_future_covariates'] = matrices
        salidas = modelo.predict_batch(
            contexts=contextos, horizon=h, use_symmetric_averaging=True,
            make_positive=True, **opciones,
        )
        return np.stack([np.asarray(s.forecast)[:h] for s in salidas])

    if ventana_pulsos is not None:
        habilitar_xreg_timesfm25(modelo)
        covariables = {nombre: [m[i].tolist() for m in matrices]
                       for i, nombre in enumerate(nombres)}
        puntos, _ = modelo.forecast_with_covariates(
            inputs=list(contextos), dynamic_numerical_covariates=covariables,
            xreg_mode='xreg + timesfm',
        )
        return np.stack([np.asarray(p)[:h] for p in puntos])

    puntos, _ = modelo.forecast(horizon=h, inputs=list(contextos))
    # Con return_backcast=True, el pronóstico está al final del array.
    return np.asarray(puntos)[:, -h:]


def pronostico_rolling_timesfm(serie, modelo, res_referencia, h=None, ventana=None,
                                 ventana_pulsos=None, verbose=True):
    """Evalúa TimesFM con o sin pulsos en los mismos orígenes del backtest."""
    h = int(res_referencia['horizonte'].max()) if h is None else h
    origenes = pd.DatetimeIndex(sorted(res_referencia['origen'].unique()))
    indices = []
    contextos = []
    for origen in origenes:
        c = serie.loc[:origen]
        c = c if ventana is None else c.iloc[-ventana:]
        indices.append(c.index)
        contextos.append(c.to_numpy(dtype=np.float32))

    t0 = time.time()
    pred = predecir_timesfm(modelo, contextos, h, indices=indices,
                            ventana_pulsos=ventana_pulsos)
    nombre = nombre_timesfm(modelo) + (' + pulsos' if ventana_pulsos is not None else '')
    if verbose:
        print(f"{nombre}: {len(origenes)} orígenes pronosticados en {time.time() - t0:.0f} s")

    filas = []
    for origen, v, c in zip(origenes, pred, contextos):
        fechas = pd.date_range(origen + pd.DateOffset(months=1), periods=h, freq='MS')
        filas.append(pd.DataFrame({
            'origen': origen, 'fecha': fechas, 'horizonte': np.arange(1, h + 1),
            'pronostico': v, 'observado': serie.reindex(fechas).to_numpy(),
            'modelo': nombre, 'n_obs': len(c),
            'n_pulsos': 0 if ventana_pulsos is None else len(pd.date_range(*ventana_pulsos, freq='MS')),
        }))
    return pd.concat(filas, ignore_index=True)


def pronostico_adelante_timesfm(serie, modelo, h=12, ventana_pulsos=None):
    """Pronostica desde el último dato disponible, con o sin pulsos."""
    origen = serie.index[-1]
    p = predecir_timesfm(modelo, [serie.to_numpy(dtype=np.float32)], h,
                         indices=[serie.index], ventana_pulsos=ventana_pulsos)[0]
    fechas = pd.date_range(origen + pd.DateOffset(months=1), periods=h, freq='MS')
    nombre = nombre_timesfm(modelo) + (' + pulsos' if ventana_pulsos is not None else '')
    return pd.DataFrame({'origen': origen, 'fecha': fechas, 'horizonte': np.arange(1, h + 1),
                         'pronostico': p, 'modelo': nombre})


def pronosticos_adelante(serie, h, modelos_timesfm=(), ventana_pulsos=None, **kwargs_arima):
    """Pronostica con benchmarks, modelos univariados y modelos con pulsos."""
    origen = serie.index[-1]
    arima = pronostico_un_origen(serie, origen, h=h, **kwargs_arima)
    partes = [pronostico_rolling_naive(serie, arima),
              pronostico_rolling_naive(serie, arima, estacional=True), arima]
    for modelo in modelos_timesfm:
        partes.append(pronostico_adelante_timesfm(serie, modelo, h))
    if ventana_pulsos is not None:
        partes.append(pronostico_un_origen(serie, origen, h=h,
                                           ventana_pulsos=ventana_pulsos, **kwargs_arima))
        for modelo in modelos_timesfm:
            partes.append(pronostico_adelante_timesfm(serie, modelo, h,
                                                       ventana_pulsos=ventana_pulsos))
    columnas = ['origen', 'fecha', 'horizonte', 'pronostico', 'modelo']
    return pd.concat([r[columnas] for r in partes], ignore_index=True)


def _errores(res):
    r = res.dropna(subset=['observado']).copy()
    r['error'] = r['observado'] - r['pronostico']
    r['error_abs'] = r['error'].abs()
    r['error_2'] = r['error'] ** 2
    return r


def metricas_por_horizonte(res):
    """Calcula RMSE, MAE, MAPE y sesgo por horizonte."""
    r = _errores(res)
    r['ape'] = r['error_abs'] / r['observado'].abs() * 100
    m = r.groupby('horizonte').agg(N=('error', 'size'), Sesgo=('error', 'mean'),
                                   MAE=('error_abs', 'mean'), MSE=('error_2', 'mean'),
                                   MAPE=('ape', 'mean'))
    m.insert(2, 'RMSE', np.sqrt(m.pop('MSE')))
    return m


def metricas_por_origen(res):
    """Calcula RMSE, MAE y sesgo para cada origen."""
    r = _errores(res)
    m = r.groupby('origen').agg(N=('error', 'size'), Sesgo=('error', 'mean'),
                                MAE=('error_abs', 'mean'), MSE=('error_2', 'mean'))
    m.insert(2, 'RMSE', np.sqrt(m.pop('MSE')))
    return m


def comparar_por_horizonte(modelos, metrica='RMSE', referencia=None):
    """Compara modelos por horizonte y contra una referencia."""
    tabla = pd.DataFrame({n: metricas_por_horizonte(r)[metrica] for n, r in modelos.items()})
    if referencia is not None:
        for n in tabla.columns.drop(referencia):
            tabla[f'{n} / {referencia}'] = tabla[n] / tabla[referencia]
    return tabla


def tabla_comparativa(modelos, metrica='RMSE', referencia='Naive', decimales=3):
    """Resume las métricas y el modelo con menor error."""
    t = pd.DataFrame({n: metricas_por_horizonte(r)[metrica] for n, r in modelos.items()})
    mejor = (t.abs() if metrica == 'Sesgo' else t).idxmin(axis=1)

    salida = t.round(decimales)
    if referencia is not None and metrica != 'Sesgo':
        ratios = t.drop(columns=referencia).div(t[referencia], axis=0).round(decimales)
        ratios.columns = [f'{n} / {referencia}' for n in ratios.columns]
        salida = pd.concat([salida, ratios], axis=1)
    salida['Mejor'] = mejor
    return salida


def errores_horizonte_fijo(res, h):
    """Extrae pronósticos y errores de un horizonte."""
    r = (res[res['horizonte'] == h].dropna(subset=['observado'])
           .set_index('fecha').sort_index())
    return pd.DataFrame({'pronostico': r['pronostico'],
                         'error': r['observado'] - r['pronostico']})


def _calc(e, metrica):
    if metrica == 'RMSE':
        return np.sqrt((e ** 2).mean())
    if metrica == 'MAE':
        return e.abs().mean()
    if metrica == 'Sesgo':
        return e.mean()
    raise ValueError("metrica debe ser 'RMSE', 'MAE' o 'Sesgo'")

PERIODOS = {
    'Prepandemia': (None, '2020-02-01'),
    'Pandemia': VENTANA_PANDEMIA,
    'Pospandemia': ('2021-10-01', None),
}


def metricas_por_periodo(modelos, h, metrica='RMSE', periodos=PERIODOS):
    """Calcula métricas a horizonte fijo por período."""
    tabla = {}
    for nombre, res in modelos.items():
        e = errores_horizonte_fijo(res, h)['error']
        tabla[nombre] = {p: _calc(e.loc[a:b], metrica) for p, (a, b) in periodos.items()}
        tabla[nombre]['Total'] = _calc(e, metrica)
    return pd.DataFrame(tabla).T


def diebold_mariano(res_a, res_b, h, perdida='cuadratica'):
    """Test de Diebold-Mariano con la corrección de Harvey-Leybourne-Newbold.

    d_t = pérdida(error de A) − pérdida(error de B), en los orígenes comunes; DM < 0: A se
    equivoca menos que B. La varianza de la media de d_t incluye autocovarianzas hasta el
    rezago h−1, porque los pronósticos a h meses de orígenes consecutivos se traslapan.
    """
    from scipy import stats

    def errores(res):
        r = res[res['horizonte'] == h].dropna(subset=['observado']).set_index('origen')
        return r['observado'] - r['pronostico']

    ea, eb = errores(res_a), errores(res_b)
    comunes = ea.index.intersection(eb.index).sort_values()
    ea, eb = ea.loc[comunes].to_numpy(), eb.loc[comunes].to_numpy()
    if perdida == 'cuadratica':
        d = ea ** 2 - eb ** 2
    elif perdida == 'absoluta':
        d = np.abs(ea) - np.abs(eb)
    else:
        raise ValueError("perdida debe ser 'cuadratica' o 'absoluta'")

    T = len(d)
    if T <= h:
        raise ValueError('Muy pocas observaciones para el horizonte solicitado')
    media = d.mean()
    dc = d - media
    gamma = [np.sum(dc[k:] * dc[:T - k]) / T for k in range(h)]
    var_media = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var_media <= 0:                       # puede ocurrir en muestras chicas: se usa solo la varianza
        var_media = gamma[0] / T
    if var_media == 0:                       # pronósticos idénticos
        return {'T': T, 'dif_media_perdida': media, 'DM': 0.0, 'p_valor': 1.0}

    correccion = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm = media / np.sqrt(var_media) * correccion
    return {'T': T, 'dif_media_perdida': media, 'DM': dm,
            'p_valor': 2 * stats.t.sf(abs(dm), df=T - 1)}


def tabla_diebold_mariano(modelos, pares, horizontes=None, perdida='cuadratica', alfa=0.05):
    """Aplica Diebold-Mariano a cada par (A, B) y horizonte, sin ajuste por comparaciones múltiples."""
    filas = []
    for a, b in pares:
        hs = horizontes if horizontes is not None else sorted(modelos[a]['horizonte'].unique())
        for h in hs:
            r = diebold_mariano(modelos[a], modelos[b], h, perdida)
            filas.append({'comparacion': f'{a} vs {b}', 'horizonte': h, **r,
                          'mejor': a if r['DM'] < 0 else b,
                          'significativo': r['p_valor'] < alfa})
    return pd.DataFrame(filas)


def resumen_diebold_mariano(tabla_dm):
    """Resume los contrastes significativos por horizonte."""
    t = tabla_dm.assign(resultado=np.where(tabla_dm['significativo'], tabla_dm['mejor'], '–'))
    orden = list(dict.fromkeys(tabla_dm['comparacion']))
    return t.pivot(index='comparacion', columns='horizonte', values='resultado').loc[orden]

COLORES = {
    'AutoARIMA': 'tab:orange',
    'AutoARIMAX': 'tab:purple',
    'TimesFM 2.5': 'tab:blue',
    'TimesFM 3.0': 'tab:green',
    'Naive': 'tab:red',
    'Naive estacional': 'tab:brown',
    'TimesFM 2.5 + pulsos': 'tab:cyan',
    'TimesFM 3.0 + pulsos': 'tab:olive',
}
_EXTRA = ['tab:blue', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray']


def _color(nombre, i=0):
    return COLORES.get(nombre, _EXTRA[i % len(_EXTRA)])


def _estilo(nombre):
    return '--' if nombre.lower().startswith('naive') else '-'


def _sombrear(ax, sombrear):
    if sombrear is not None:
        ax.axvspan(pd.Timestamp(sombrear[0]), pd.Timestamp(sombrear[1]), color='gray', alpha=0.12)


def _dibujar_rolling(ax, serie, res, alpha=0.25, desde=None):
    desde = pd.Timestamp(desde) if desde is not None else res['origen'].min()
    for origen, g in res.groupby('origen'):
        x = [origen] + g['fecha'].tolist()
        y = [serie.loc[origen]] + g['pronostico'].tolist()
        ax.plot(x, y, color='gray', alpha=alpha, linewidth=1)
    obs = serie.loc[desde:res['fecha'].max()]
    ax.plot(obs.index, obs.values, color='black', linewidth=2.2, zorder=10, label='Observado')
    ax.plot([], [], color='gray', alpha=0.7, linewidth=1,
            label=f"Pronósticos a {res['horizonte'].max()} meses ({res['origen'].nunique()} orígenes)")
    ax.grid(alpha=0.3)


def graficar_series(series, titulo='Tasa de desempleo', ylabel='Tasa de desempleo (%)',
                    sombrear=None, figsize=(13, 5)):
    """Grafica una o más series."""
    fig, ax = plt.subplots(figsize=figsize)
    for i, (nombre, s) in enumerate(series.items()):
        ax.plot(s.index, s.values, linewidth=2, label=nombre,
                color='black' if i == 0 else _EXTRA[(i - 1) % len(_EXTRA)])
    _sombrear(ax, sombrear)
    ax.set_title(titulo, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.show()
    return fig, ax


def graficar_adelante(series, adelante, desde=None, ylabel='Tasa de desempleo (%)', alto_panel=4.5):
    """Grafica los pronósticos posteriores al último dato."""
    n = len(series)
    fig, ejes = plt.subplots(n, 1, figsize=(13, alto_panel * n), squeeze=False)
    for ax, (nombre, s) in zip(ejes[:, 0], series.items()):
        obs = s.loc[desde:] if desde is not None else s.iloc[-36:]
        ax.plot(obs.index, obs.values, color='black', linewidth=2.2, label='Observado', zorder=10)
        d = adelante[adelante['region'] == nombre]
        for i, (modelo, g) in enumerate(d.groupby('modelo', sort=False)):
            x = [s.index[-1]] + g['fecha'].tolist()
            y = [s.iloc[-1]] + g['pronostico'].tolist()
            ax.plot(x, y, color=_color(modelo, i), linestyle=_estilo(modelo), linewidth=1.8,
                    marker='o', markersize=3, label=modelo)
        ax.axvline(s.index[-1], color='gray', linestyle=':', linewidth=1)
        ax.set_title(nombre, fontweight='bold', loc='left')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=8, ncol=3)
    h = adelante['horizonte'].max()
    fig.suptitle(f'Pronóstico a {h} meses desde el último dato', fontweight='bold', fontsize=13)
    fig.tight_layout()
    plt.show()
    return fig, ejes


def graficar_rolling(serie, res, desde=None, alpha=0.25, figsize=(13, 5),
                     ylabel='Tasa de desempleo (%)', titulo=None, ylim=None, sombrear=None):
    """Grafica los pronósticos rolling de un modelo."""
    fig, ax = plt.subplots(figsize=figsize)
    _dibujar_rolling(ax, serie, res, alpha=alpha, desde=desde)
    _sombrear(ax, sombrear)
    ax.set_title(titulo or f"Validación rolling: pronósticos a {res['horizonte'].max()} meses vs. observado",
                 fontweight='bold')
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.show()
    return fig, ax


def graficar_modelo(serie, res, nombre=None, referencia=None, referencias=None, desde=None, alpha=0.25,
                    ylim=None, ylabel='Tasa de desempleo (%)', figsize=(13, 10)):
    """Grafica pronósticos rolling, RMSE y sesgo por origen."""
    nombre = nombre or res['modelo'].iloc[0]
    h = res['horizonte'].max()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True,
                                        gridspec_kw={'height_ratios': [2, 1, 1]})

    _dibujar_rolling(ax1, serie, res, alpha=alpha, desde=desde)
    ax1.set_title(nombre, fontweight='bold', loc='left')
    ax1.set_ylabel(ylabel)
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.legend(loc='upper left')

    curvas = [(nombre, res)]
    if referencia is not None:
        curvas.append((referencia['modelo'].iloc[0], referencia))
    if referencias is not None:
        curvas.extend((n, r) for n, r in referencias.items() if n != nombre)
    for i, (n, r) in enumerate(curvas):
        m = metricas_por_origen(r)
        estilo = dict(color=_color(n, i), linestyle=_estilo(n), linewidth=1.5,
                      marker='o', markersize=3, label=n)
        ax2.plot(m.index, m['RMSE'], **estilo)
        ax3.plot(m.index, m['Sesgo'], **estilo)

    ax2.set_ylabel(f'RMSE (1 a {h} meses)')
    ax3.set_ylabel('Sesgo (obs − pron)')
    ax3.axhline(0, color='gray', linewidth=1)
    ax3.set_xlabel('Fecha (en paneles 2 y 3: origen del pronóstico)')
    ax2.legend(loc='upper left', ncol=min(3, len(curvas)), fontsize=9)
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)

    fig.suptitle(f'{nombre}: pronósticos rolling a {h} meses', fontweight='bold', fontsize=13)
    fig.tight_layout()
    plt.show()
    return fig, (ax1, ax2, ax3)


def graficar_rolling_modelos(serie, modelos, desde=None, alpha=0.25, alto_panel=4,
                             ylabel='Tasa de desempleo (%)', ylim=None, sombrear=None):
    """Grafica los pronósticos rolling de varios modelos."""
    n = len(modelos)
    fig, ejes = plt.subplots(n, 1, figsize=(13, alto_panel * n), sharex=True, sharey=True, squeeze=False)
    for ax, (nombre, res) in zip(ejes[:, 0], modelos.items()):
        _dibujar_rolling(ax, serie, res, alpha=alpha, desde=desde)
        _sombrear(ax, sombrear)
        ax.set_title(nombre, fontweight='bold', loc='left')
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ejes[0, 0].set_ylim(*ylim)
    ejes[0, 0].legend(loc='upper left')
    h = max(r['horizonte'].max() for r in modelos.values())
    fig.suptitle(f'Validación rolling: pronósticos a {h} meses vs. observado', fontweight='bold', fontsize=13)
    fig.tight_layout()
    plt.show()
    return fig, ejes


def _graficar_metrica(modelos, tabla_fn, eje_x, metrica, figsize, titulo, ylim, xlabel, ticks=None):
    fig, ax = plt.subplots(figsize=figsize)
    for i, (nombre, res) in enumerate(modelos.items()):
        m = tabla_fn(res)
        ax.plot(m.index, m[metrica], color=_color(nombre, i), linestyle=_estilo(nombre),
                linewidth=2 if eje_x == 'horizonte' else 1.3, marker='o',
                markersize=4 if eje_x == 'horizonte' else 2.5, label=nombre)
    if metrica == 'Sesgo':
        ax.axhline(0, color='gray', linewidth=1)
        ax.set_ylabel('Sesgo (observado − pronóstico)')
    else:
        ax.set_ylabel(metrica)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ticks is not None:
        ax.set_xticks(ticks)
    ax.set_xlabel(xlabel)
    ax.set_title(titulo, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.show()
    return fig, ax


def graficar_metrica_por_horizonte(modelos, metrica='RMSE', figsize=(10, 5), titulo=None, ylim=None):
    """Grafica una métrica por horizonte."""
    h = max(r['horizonte'].max() for r in modelos.values())
    return _graficar_metrica(modelos, metricas_por_horizonte, 'horizonte', metrica, figsize,
                             titulo or f'{metrica} por horizonte de pronóstico', ylim,
                             'Horizonte (meses)', ticks=range(1, h + 1))


def graficar_comparacion(modelos, metricas=('RMSE', 'Sesgo'), figsize=(14, 5)):
    """Compara las métricas por horizonte de varios modelos."""
    fig, ejes = plt.subplots(1, len(metricas), figsize=figsize, squeeze=False)
    h = max(r['horizonte'].max() for r in modelos.values())
    for ax, metrica in zip(ejes[0], metricas):
        for i, (nombre, res) in enumerate(modelos.items()):
            m = metricas_por_horizonte(res)
            ax.plot(m.index, m[metrica], color=_color(nombre, i), linestyle=_estilo(nombre),
                    linewidth=2, marker='o', markersize=4, label=nombre)
        if metrica == 'Sesgo':
            ax.axhline(0, color='gray', linewidth=1)
            ax.set_title('Sesgo (observado − pronóstico)', fontweight='bold')
        else:
            ax.set_title(metrica, fontweight='bold')
        ax.set_xticks(range(1, h + 1))
        ax.set_xlabel('Horizonte (meses)')
        ax.grid(alpha=0.3)
    manejadores, etiquetas = ejes[0, 0].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc='upper center', ncol=len(modelos), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()
    return fig, ejes


def graficar_metrica_por_origen(modelos, metrica='RMSE', figsize=(13, 4), titulo=None, ylim=None):
    """Grafica una métrica por origen."""
    h = max(r['horizonte'].max() for r in modelos.values())
    return _graficar_metrica(modelos, metricas_por_origen, 'origen', metrica, figsize,
                             titulo or f'{metrica} por origen del pronóstico (horizonte 1 a {h} meses)',
                             ylim, 'Origen del pronóstico')


def graficar_horizonte_fijo(serie, modelos, h=1, ventana_rmse=6, sombrear=None, modelos_rolling=None,
                            marcar_h=True, alpha=0.25, ylim=None, ylim_error=None, ylim_rmse=None,
                            ylabel='Tasa de desempleo (%)', alto_panel=3):
    """Grafica errores y RMSE móvil a horizonte fijo."""
    if modelos_rolling is None:
        modelos_rolling = [n for n in modelos if not n.lower().startswith('naive')]
    n = len(modelos_rolling)
    fig, ejes = plt.subplots(n + 2, 1, figsize=(13, alto_panel * n + 6), sharex=True,
                             gridspec_kw={'height_ratios': [1.3] * n + [1, 1]})
    ejes_rolling, ax_err, ax_rmse = ejes[:n], ejes[n], ejes[n + 1]

    for i, (ax, nombre) in enumerate(zip(ejes_rolling, modelos_rolling)):
        res = modelos[nombre]
        _dibujar_rolling(ax, serie, res, alpha=alpha)
        if marcar_h:
            d = errores_horizonte_fijo(res, h)
            ax.scatter(d.index, d['pronostico'], s=10, color=_color(nombre, i), zorder=11,
                       label=f'Pronóstico a {h} {"mes" if h == 1 else "meses"}')
        ax.set_title(nombre, fontweight='bold', loc='left')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper left', fontsize=8)
        if i > 0:
            ax.sharey(ejes_rolling[0])
    if ylim is not None:
        ejes_rolling[0].set_ylim(*ylim)

    for i, (nombre, res) in enumerate(modelos.items()):
        d = errores_horizonte_fijo(res, h)
        estilo = dict(color=_color(nombre, i), linestyle=_estilo(nombre), linewidth=1.6, label=nombre)
        ax_err.plot(d.index, d['error'], **estilo)
        rmse = np.sqrt((d['error'] ** 2).rolling(ventana_rmse, min_periods=ventana_rmse).mean())
        ax_rmse.plot(rmse.index, rmse, **estilo)

    ax_err.axhline(0, color='black', linewidth=1)
    ax_err.set_ylabel(f'Error a {h}m (obs − pron)')
    ax_rmse.set_ylabel(f'RMSE móvil {ventana_rmse}m')
    ax_rmse.set_xlabel('Fecha')
    ax_err.legend(loc='upper left', ncol=len(modelos), fontsize=8)
    for ax, lim in ((ax_err, ylim_error), (ax_rmse, ylim_rmse)):
        if lim is not None:
            ax.set_ylim(*lim)
    for ax in ejes:
        _sombrear(ax, sombrear)
        ax.grid(alpha=0.3)

    fig.suptitle(f'Pronósticos rolling y desempeño a {h} {"mes" if h == 1 else "meses"}',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    plt.show()
    return fig, ejes


def _archivo(nombre):
    return nombre.lower().replace(' ', '_').replace('.', '') + '.csv'



def guardar_resultados(modelos, carpeta='resultados'):
    """Guarda los resultados de cada modelo como CSV."""
    os.makedirs(carpeta, exist_ok=True)
    for nombre, res in modelos.items():
        res.assign(modelo=nombre).to_csv(os.path.join(carpeta, _archivo(nombre)), index=False)
    print(f'{len(modelos)} archivos guardados en {carpeta}/')


def cargar_resultados(carpeta='resultados'):
    """Carga únicamente los CSV de resultados del backtest."""
    modelos = {}
    columnas = {'origen', 'fecha', 'horizonte', 'observado', 'pronostico', 'modelo'}
    for archivo in sorted(os.listdir(carpeta)):
        if not archivo.endswith('.csv'):
            continue
        res = pd.read_csv(os.path.join(carpeta, archivo))
        if res.empty or not columnas.issubset(res.columns):
            continue
        res[['origen', 'fecha']] = res[['origen', 'fecha']].apply(pd.to_datetime)
        modelos[res['modelo'].iloc[0]] = res
    return modelos
