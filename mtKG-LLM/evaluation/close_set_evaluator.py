class CloseSetEvaluator:
    def __init__(self):
        self.correct_num = dict()
        self.all_num = dict()
        self.precise_num = dict()
        self.predict_num = dict()

    def add_data(self, pred, ground_truth):
        self.all_num[ground_truth] = self.all_num.get(ground_truth, 0) + 1
        self.predict_num[pred] = self.predict_num.get(pred, 0) + 1
        if pred == ground_truth:
            self.correct_num[ground_truth] = self.correct_num.get(ground_truth, 0) + 1
            self.precise_num[pred] = self.precise_num.get(pred, 0) + 1

    def get_accuracy(self):
        result = dict()
        for key, all_num in self.all_num.items():
            result[key] = self.correct_num.get(key, 0) / all_num
        return result
    
    def get_mean_precision(self):
        result = dict()
        for key, pred_num in self.predict_num.items():
            result[key] = self.precise_num.get(key, 0) / pred_num
        
        sum = 0
        for key in ['Leader-Sub', 'Colleague', 'Service', 'Parent-offs', 'Sibling', 'Couple', 'Friend', 'Opponent']:
            sum += result[key]
        return sum / 8
