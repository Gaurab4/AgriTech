import numpy as np
from transformers import pipeline
from bs4 import BeautifulSoup
import requests


def recommendation(N, P, k, temperature, humidity, ph, rainfal):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfal]])
    transformed_features = ms.fit_transform(features)
    transformed_features = sc.fit_transform(transformed_features)
    prediction = rfc.predict(transformed_features).reshape(1, -1)

    return prediction[0]
