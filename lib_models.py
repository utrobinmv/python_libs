class MyLinearRegression:
    '''
    Линейная регрессия
    Линейные методы предполагают, что между признаками объекта (features) и целевой переменной (target/label) существует линейная зависимость, то есть
    '''
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # Принимает на вход X, y и вычисляет веса по данной выборке
        # Не забудьте про фиктивный признак равный 1
        
        n, k = X.shape
        
        X_train = X
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))

        self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y

        return self
        
    def predict(self, X):
        # Принимает на вход X и возвращает ответы модели
        # Не забудьте про фиктивный признак равный 1
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))

        y_pred = X_train @ self.w

        return y_pred
    
    def get_weights(self):
        return self.w



def train_MyLinearRegression(X_train, y_train, X_test):
    '''
    Обучает данные через алгоритм линейной регрессии
    '''
    regressor = MyLinearRegression()

    regressor.fit(X_train[:, np.newaxis], y_train)

    predictions = regressor.predict(X_test[:, np.newaxis])
    w = regressor.get_weights()
    print(w)
    print(predictions)


class MyGradientLinearRegression(MyLinearRegression):
    '''
    Градиентная оптимизация
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # передает именные параметры родительскому конструктору
        self.w = None
    
    def fit(self, X, y, lr=0.01, max_iter=100):
        # Принимает на вход X, y и вычисляет веса по данной выборке
        # Не забудьте про фиктивный признак равный 1!

        n, k = X.shape

        # случайно инициализируем веса
        if self.w is None:
            self.w = np.random.randn(k + 1 if self.fit_intercept else k)
        
        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X
        
        self.losses = []
        
        for iter_num in range(max_iter):
            y_pred = self.predict(X)
            self.losses.append(mean_squared_error(y_pred, y))

            grad = self._calc_gradient(X_train, y, y_pred)

            assert grad.shape == self.w.shape, f"gradient shape {grad.shape} is not equal weight shape {self.w.shape}"
            self.w -= lr * grad

        return self

    def _calc_gradient(self, X, y, y_pred):
        grad = 2 * (y_pred - y)[:, np.newaxis] * X
        grad = grad.mean(axis=0)
        return grad

    def get_losses(self):
        return self.losses

def train_MyGradientLinearRegression(X_train, y_train, X_test):
    '''
    Обучает данные через алгоритм Градиентная оптимизация
    '''
    regressor = MyGradientLinearRegression(fit_intercept=True)

    l = regressor.fit(X_train[:, np.newaxis], y_train, max_iter=100).get_losses()

    predictions = regressor.predict(X_test[:, np.newaxis])
    w = regressor.get_weights()
    
    
    
class MySGDLinearRegression(MyGradientLinearRegression):
    '''
    Стохатистический градиентный спуск
    '''
    def __init__(self, n_sample=10, **kwargs):
        super().__init__(**kwargs) # передает именные параметры родительскому конструктору
        self.w = None
        self.n_sample = n_sample

    def _calc_gradient(self, X, y, y_pred):
        # Главное отличие в SGD - это использование подвыборки для шага оптимизации
        inds = np.random.choice(np.arange(X.shape[0]), size=self.n_sample, replace=False)
        
        grad = 2 * (y_pred[inds] - y[inds])[:, np.newaxis] * X[inds]
        grad = grad.mean(axis=0)

        return grad
        
def train_MySGDLinearRegression(X_train, y_train, X_test):
    '''
    Обучает данные через алгоритм Стохатистический градиентный спуск
    '''    

    regressor = MySGDLinearRegression(fit_intercept=True)

    l = regressor.fit(X_train[:, np.newaxis], y_train, max_iter=100).get_losses()

    predictions = regressor.predict(X_test[:, np.newaxis])
    w = regressor.get_weights()
    

def logit(x, w):
    return np.dot(x, w)

def sigmoid(h):
    return 1. / (1 + np.exp(-h))

class MyLogisticRegression(object):
    '''
    Логистическая регрессия
    '''
    def __init__(self):
        self.w = None
    
    def fit(self, X, y, max_iter=100, lr=0.1):
        # Принимает на вход X, y и вычисляет веса по данной выборке.
        # Множество допустимых классов: {1, -1}
        # Не забудьте про фиктивный признак равный 1!
        
        n, k = X.shape
        
        if self.w is None:
            self.w = np.random.randn(k + 1)
        
        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)
        
        losses = []
        
        for iter_num in range(max_iter):
            z = sigmoid(logit(X_train, self.w))
            grad = np.dot(X_train.T, (z - y)) / len(y)

            self.w -= grad * lr

            losses.append(self.__loss(y, z))
        
        return losses
        
    def predict_proba(self, X):
        # Принимает на вход X и возвращает ответы модели
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logit(X_, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold
    
    def get_weights(self):
        return self.w
      
    def __loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def train_MyLogisticRegression(X, y, X_test):
    '''
    Обучает данные через алгоритм Логистической регрессии
    '''    

    clf = MyLogisticRegression()

    clf.fit(X, y, max_iter=1000)

    w = clf.get_weights()

    Z = clf.predict(X_test)
    
    Z2 = clf.predict_proba(X_test)
