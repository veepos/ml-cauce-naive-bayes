"""Naive Bayes multinomial implementado desde cero.

Reglas:
- Los cálculos se hacen solo con NumPy. No se puede importar scipy ni usar sklearn.naive_bayes.
- La clase hereda de ClassifierMixin y BaseEstimator solo para poder usarse dentro de un
  Pipeline y de GridSearchCV de scikit-learn (eso le da get_params, set_params y score).
- Todo se calcula en escala logarítmica: multiplicar cientos de probabilidades chicas da 0
  en punto flotante (underflow).
- X puede ser un array de NumPy o una matriz dispersa de SciPy (la que devuelve CountVectorizer).
  Las operaciones X[filas], X.sum(axis=0) y X @ M funcionan con ambas; usá np.asarray(...) para
  convertir los resultados a arrays comunes.

Verificá tu trabajo con:  python -m pytest -q tests/test_naive_bayes.py
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _validar_matriz_conteos(X, n_features=None):
    """Valida la forma y los valores de una matriz densa o dispersa de conteos."""
    if not hasattr(X, "shape") or len(X.shape) != 2 or 0 in X.shape:
        raise ValueError("X debe ser una matriz no vacía de documentos y términos.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X debe tener {n_features} términos, como los datos de entrenamiento.")

    valores = X if isinstance(X, np.ndarray) else getattr(X, "data", None)
    if valores is None:
        raise ValueError("X debe ser un array de NumPy o una matriz dispersa numérica.")
    try:
        validos = np.isfinite(valores).all() and (valores >= 0).all()
    except (TypeError, ValueError):
        validos = False
    if not validos:
        raise ValueError("Los conteos deben ser numéricos, finitos y no negativos.")


def log_suma_exp(a, axis=1):
    """Calcula log(sum(exp(a))) a lo largo de `axis` sin overflow ni underflow.

    Truco: si m es el máximo de a, entonces log(sum(exp(a))) = m + log(sum(exp(a - m))).

    Ejemplo: log_suma_exp(np.array([[-1000.0, -1000.0]])) -> array([-999.3069...])
    """
    a = np.asarray(a, dtype=float)
    maximo = np.max(a, axis=axis, keepdims=True)
    # Si toda una fila es -inf, restar -inf produciría NaN.
    desplazamiento = np.where(np.isfinite(maximo), maximo, 0.0)
    with np.errstate(divide="ignore", under="ignore"):
        resultado = desplazamiento + np.log(np.exp(a - desplazamiento).sum(axis=axis, keepdims=True))
    return np.squeeze(resultado, axis=axis)


class NaiveBayesMultinomial(ClassifierMixin, BaseEstimator):
    """Naive Bayes multinomial con suavizado aditivo.

    Parámetros
    ----------
    alpha : float, por defecto 1.0
        Suavizado aditivo. Con alpha=1 es el suavizado de Laplace.

    Atributos que tiene que crear fit (mismos nombres que en scikit-learn)
    -------------------------------------------------------------------
    classes_ : array (n_clases,)
        Clases ordenadas, como devuelve np.unique(y).
    class_count_ : array (n_clases,)
        Cantidad de documentos de cada clase.
    feature_count_ : array (n_clases, n_terminos)
        Suma de los conteos de cada término en los documentos de cada clase.
    class_log_prior_ : array (n_clases,)
        log P(c) = log(documentos de la clase c / documentos totales).
    feature_log_prob_ : array (n_clases, n_terminos)
        log P(t | c) = log((conteo de t en c + alpha) / (total de términos en c + alpha * n_terminos)).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        """Estima las probabilidades a partir de la matriz de conteos X y las etiquetas y. Devuelve self."""
        y = np.asarray(y)
        _validar_matriz_conteos(X)
        if y.ndim != 1 or X.shape[0] != len(y):
            raise ValueError("X debe tener documentos y términos, con una etiqueta por documento.")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha debe ser un número finito no negativo.")

        self.classes_, indices, conteos = np.unique(y, return_inverse=True, return_counts=True)
        self.class_count_ = conteos.astype(float)
        self.n_features_in_ = X.shape[1]
        self.feature_count_ = np.zeros((len(self.classes_), self.n_features_in_))
        for indice in range(len(self.classes_)):
            # Sólo se vuelve denso el total de cada clase, no la matriz de documentos.
            self.feature_count_[indice] = np.asarray(X[indices == indice].sum(axis=0)).ravel()

        self.class_log_prior_ = np.log(self.class_count_ / len(y))
        suavizados = self.feature_count_ + self.alpha
        totales = suavizados.sum(axis=1, keepdims=True)
        if (totales == 0).any():
            raise ValueError("Una clase no tiene términos: usá alpha positivo para suavizarla.")
        with np.errstate(divide="ignore"):
            self.feature_log_prob_ = np.log(suavizados) - np.log(totales)
        return self

    def _log_verosimilitud_conjunta(self, X):
        """Devuelve log P(c) + sum_t x_t · log P(t | c) para cada documento y clase.

        Forma del resultado: (n_documentos, n_clases). Se resuelve con un solo producto matricial.
        """
        if not hasattr(self, "feature_log_prob_"):
            raise ValueError("Primero entrená el clasificador con fit.")
        _validar_matriz_conteos(X, self.n_features_in_)
        if self.alpha == 0:
            # Un término ausente no aporta; uno observado con P(t|c)=0 descarta la clase.
            ceros = np.isneginf(self.feature_log_prob_)
            log_prob = np.where(ceros, 0.0, self.feature_log_prob_)
            puntajes = np.asarray(X @ log_prob.T) + self.class_log_prior_
            puntajes[np.asarray(X @ ceros.T) > 0] = -np.inf
            return puntajes
        return np.asarray(X @ self.feature_log_prob_.T) + self.class_log_prior_

    def predict_log_proba(self, X):
        """log P(c | documento): la log-verosimilitud conjunta normalizada con log_suma_exp."""
        puntajes = self._log_verosimilitud_conjunta(X)
        normalizacion = log_suma_exp(puntajes, axis=1)
        if np.isneginf(normalizacion).any():
            raise ValueError("El mensaje tiene probabilidad cero en todas las clases: usá alpha positivo.")
        return puntajes - normalizacion[:, np.newaxis]

    def predict_proba(self, X):
        """P(c | documento). Cada fila suma 1. Las columnas siguen el orden de classes_."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """La clase con mayor probabilidad a posteriori para cada documento."""
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


def terminos_mas_indicativos(modelo, vocabulario, n=10):
    """Devuelve {clase: [n términos]} con los términos que más empujan hacia cada clase.

    Para la clase c, el puntaje de un término t es
        log P(t | c) - promedio de log P(t | c') en las demás clases c'.
    Se devuelven los n términos de mayor puntaje, de mayor a menor.

    `vocabulario` es la lista de términos en el orden de las columnas de X
    (por ejemplo, CountVectorizer.get_feature_names_out()).
    """
    vocabulario = np.asarray(vocabulario)
    log_prob = modelo.feature_log_prob_
    if len(modelo.classes_) < 2:
        raise ValueError("Se necesitan al menos dos clases para comparar términos.")
    if vocabulario.ndim != 1 or len(vocabulario) != log_prob.shape[1]:
        raise ValueError("El vocabulario debe coincidir con las columnas del modelo.")
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n debe ser un entero no negativo.")

    resultado = {}
    for indice, clase in enumerate(modelo.classes_):
        otras_clases = np.arange(len(modelo.classes_)) != indice
        puntajes = log_prob[indice] - log_prob[otras_clases].mean(axis=0)
        orden = np.argsort(-puntajes, kind="stable")[:n]
        resultado[clase] = vocabulario[orden].tolist()
    return resultado
