from SurvTRACE.survtrace.evaluate_utils import Evaluator


class Custom_Evaluation(Evaluator):
    def __int__(self, df, train_index):
        super().__init__(df, train_index)

    def eval(self, model, test_set, confidence=None, val_batch_size=None):
        return None
