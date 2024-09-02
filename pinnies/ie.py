from .base import BaseIE


class IE1D(BaseIE):
    def __init__(self, domain, num_train):
        self.domain = [domain]
        self.N = num_train
        self.x = None
        self.weights = None
        self.X = None
        self.T = None


class IE2D(BaseIE):
    def __init__(self, domain, num_train):
        self.domain = domain
        self.N = num_train
        self.x = None
        self.weights = None
        self.X = None
        self.Y = None
        self.S = None
        self.T = None


class IE3D(BaseIE):
    def __init__(self, domain, num_train):
        self.domain = domain
        self.N = num_train
        self.x = None
        self.weights = None
        self.X = None
        self.Y = None
        self.Z = None
        self.R = None
        self.S = None
        self.T = None
