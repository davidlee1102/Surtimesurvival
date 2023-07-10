from collections import defaultdict

from SurvTRACE.survtrace.evaluate_utils import Evaluator

import numpy as np


class Custom_Evaluation(Evaluator):
    def __int__(self, df, train_index):
        super().__init__(df, train_index)

    def eval(self, model, test_set, confidence=None, val_batch_size=None, our_mask=None):
        '''do evaluation.
        if confidence is not None, it should be in (0, 1) and the confidence
        interval will be given by bootstrapping.
        '''
        print("***" * 10)
        print("start evaluation")
        print("***" * 10)

        if confidence is None:
            if model.config['num_event'] > 1:
                print("Evaluate with multiple event")
                return self.eval_multi(model, test_set, val_batch_size, our_mask)
            else:
                print("Evaluate with single event")
                return self.eval_single(model, test_set, val_batch_size, our_mask)

        else:

            print("Confidence is not supported at the moment")
            return None
