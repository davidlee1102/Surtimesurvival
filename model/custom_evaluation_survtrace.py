from collections import defaultdict

from SurvTRACE.survtrace.evaluate_utils import Evaluator
from sksurv.metrics import concordance_index_ipcw, brier_score

import numpy as np


class Custom_Evaluation:
    def __init__(self, df, train_index):
        '''the input duration_train should be the raw durations (continuous),
        NOT the discrete index of duration.
        '''
        self.df_train_all = df.loc[train_index]

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

    def eval_single(self, model, test_set, val_batch_size=None, our_mask=None):

        df_train_all = self.df_train_all
        get_target = lambda df: (df['duration'].values, df['event'].values)
        durations_train, events_train = get_target(df_train_all)
        et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                            dtype=[('e', bool), ('t', float)])
        times = model.config['duration_index'][1:-1]
        horizons = model.config['horizons']

        df_test, df_y_test = test_set
        surv = model.predict_surv(df_test, batch_size=val_batch_size, our_mask=our_mask)
        risk = 1 - surv

        durations_test, events_test = get_target(df_y_test)
        et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                           dtype=[('e', bool), ('t', float)])

        metric_dict = defaultdict(list)

        brs = brier_score(et_train, et_test, surv.to("cpu").numpy()[:, 1:-1], times)[1]
        cis = []

        for i, _ in enumerate(times):
            cis.append(
                concordance_index_ipcw(et_train, et_test, estimate=risk[:, i + 1].to("cpu").numpy(), tau=times[i])[0]
            )
            metric_dict[f'{horizons[i]}_ipcw'] = cis[i]
            metric_dict[f'{horizons[i]}_brier'] = brs[i]

        for horizon in enumerate(horizons):
            print(f"For {horizon[1]} quantile,")
            print("TD Concordance Index - IPCW:", cis[horizon[0]])
            print("Brier Score:", brs[horizon[0]])

        return metric_dict
