import sklearn
import sklearn.metrics
from enums import metric_enums

class Metrics:
    def __init__(self, metric_name: str):
        self.metric_name = metric_name

    def get_accuracy_score(self, targets, outputs):
        return sklearn.metrics.accuracy_score(targets, outputs)

    def get_f1_score(self, targets, outputs, average='macro'):
        AVERAGE_LIST = ['binary', 'micro', 'macro', 'weighted', 'samples']
        assert average in AVERAGE_LIST, f"{average} not in F1-score metrics list i.e. {AVERAGE_LIST}"  
        return sklearn.metrics.f1_score(targets, outputs, average=average)

    def get_metrics_fn(self):
        metrics_fn = {metric_enums.MetricsEnum.ACCURACY_SCORE.value: self.get_accuracy_score,
                      metric_enums.MetricsEnum.F1_SCORE.value: self.get_f1_score}
        
        return metrics_fn[self.metric_name]
    
if __name__ == "__main__":
    metrics = Metrics("f1_score")
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    print(metrics.get_f1_score(y_true, y_pred))
    print(metrics.get_f1_score(y_true, y_pred, average='micro'))
    print(metrics.get_f1_score(y_true, y_pred, average='weighted'))

    print("\n### TEST Function Wrapper ###")
    print(metrics.get_metrics_fn()(y_true, y_pred))

