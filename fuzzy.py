import numpy as np

class Fuzzy:
    @staticmethod
    def trapmf(x, a, b, c, d):
        x = np.asarray(x)
        y = np.zeros_like(x, dtype=float)
        
        mask = (a < x) & (x <= b)
        y[mask] = (x[mask] - a) / (b - a)
        mask = (b < x) & (x <= c)
        y[mask] = 1.0
        mask = (c < x) & (x < d)
        y[mask] = (d - x[mask]) / (d - c)
        
        return y

    @staticmethod
    def gaussmf(x, c, sigma):
        x = np.asarray(x)
        return np.exp(-0.5 * ((x - c) / sigma) ** 2)
    @staticmethod
    def calculate_trapmf(change, a, b, c, d):
        return Fuzzy.trapmf(np.array(change), a, b, c, d)
    @staticmethod
    def calculate_gaussmf(change, c, sigma):
        return Fuzzy.gaussmf(np.array(change), c, sigma)
    @staticmethod
    def interp_membership(change, membership, deltaP):
        change = np.array(change)
        membership = np.array(membership)
        idx = np.searchsorted(change, deltaP)
    
        if idx == 0 or idx == len(change):
            return 0.0
            
        x0, x1 = change[idx-1], change[idx]
        y0, y1 = membership[idx-1], membership[idx]
        
        return y0 + (deltaP - x0) * (y1 - y0) / (x1 - x0)

    @staticmethod
    def defuzz_centroid(x, mu):
        x = np.array(x)
        mu = np.array(mu)
        
        numerator = np.sum(x * mu)
        denominator = np.sum(mu)
        
        return numerator / denominator if denominator != 0 else 0.0

    @staticmethod
    def fmin(value, array):
        return np.minimum(value, np.array(array))

    @staticmethod
    def fmax(first_arr, second_arr):
        return np.maximum(np.array(first_arr), np.array(second_arr))

    @staticmethod
    def all_zero(arr):
        return np.all(np.array(arr) == 0.0) 