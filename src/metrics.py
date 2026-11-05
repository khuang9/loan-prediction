from sklearn.metrics import fbeta_score

def build_f2_scorer(threshold):
    def f2_scorer(model, X, y):
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        return fbeta_score(y, y_pred, beta=2)
    return f2_scorer
    