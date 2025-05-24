import ruptures as rpt

class ChangePointDetector:
    def __init__(self):
        self.model = rpt.Pelt(model="rbf")

    def score(self, series):
        pts = self.model.fit_predict(series.values)
        return len(pts)