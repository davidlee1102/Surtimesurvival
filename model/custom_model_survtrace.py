from SurvTRACE.survtrace.utils import set_random_seed
from SurvTRACE.survtrace.model import SurvTraceSingle
from SurvTRACE.survtrace.config import STConfig
from SurvTRACE.survtrace.utils import pad_col

import torch
import numpy as np
import pandas as pd

import torch.nn.functional as F
from model.survtimesurvival_model import TransformerClassifier

# define the setup parameters
STConfig['data'] = 'metabric'

set_random_seed(STConfig['seed'])

hparams = {
    'batch_size': 2,
    'weight_decay': 1e-4,
    'learning_rate': 1e-3,
    'epochs': 1,
}

model = SurvTraceSingle(STConfig)


class Custom_SurvTrace(SurvTraceSingle):
    def __init__(self, config):
        super().__init__(config)
        self.our_model = TransformerClassifier(input_dim=21, seq_length=16, embed_dim=64, num_heads=2,
                                               ffn_hidden_dim=64, num_layers=2)

    def forward(
            self,
            input_ids=None,
            input_nums=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            our_mask=None,
            **kwargs
    ):

        output = self.our_model(input_nums, our_mask)
        x = super().forward(input_ids=output[:, :0].long(), input_nums=output)
        return x

    def predict(self, x_input, batch_size=None, our_mask=None):
        if not isinstance(x_input, torch.Tensor):
            x_input_cat = x_input.iloc[:, :self.config.num_categorical_feature]
            x_input_num = x_input.iloc[:, self.config.num_categorical_feature:]
            x_num = torch.tensor(x_input_num.values).float()
            x_cat = torch.tensor(x_input_cat.values).long()
        else:
            x_cat = x_input[:, :self.config.num_categorical_feature].long()
            x_num = x_input[:, self.config.num_categorical_feature:].float()

        if self.use_gpu:
            x_num = x_num.cuda()
            x_cat = x_cat.cuda()
            our_mask = our_mask.cuda()

        num_sample = len(x_num)
        self.eval()
        with torch.no_grad():
            if batch_size is None:
                preds = self.forward(input_ids=x_cat, input_nums=x_num, our_mask=our_mask)[1]
            else:
                preds = []
                num_batch = int(np.ceil(num_sample / batch_size))
                for idx in range(num_batch):
                    batch_x_num = x_num[idx * batch_size:(idx + 1) * batch_size]
                    batch_x_cat = x_cat[idx * batch_size:(idx + 1) * batch_size]
                    batch_our_mask = our_mask[idx * batch_size:(idx + 1) * batch_size]

                    batch_pred = self.forward(input_ids=batch_x_cat, input_nums=batch_x_num, our_mask=batch_our_mask)
                    preds.append(batch_pred[1])
                preds = torch.cat(preds)
        return preds

    def predict_hazard(self, input_ids, batch_size=None, our_mask=None):
        preds = self.predict(input_ids, batch_size, our_mask)
        hazard = F.softplus(preds)
        hazard = pad_col(hazard, where="start")
        return hazard

    def predict_risk(self, input_ids, batch_size=None, our_mask=None):
        surv = self.predict_surv(input_ids, batch_size, our_mask)
        return 1 - surv

    def predict_surv(self, input_ids, batch_size=None, our_mask=None, epsilon=1e-7):
        hazard = self.predict_hazard(input_ids, batch_size, our_mask)
        # surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()
        surv = hazard.cumsum(1).mul(-1).exp()
        return surv

    def predict_surv_df(self, input_ids, batch_size=None, our_mask=None):
        surv = self.predict_surv(input_ids, batch_size, our_mask)
        return pd.DataFrame(surv.to("cpu").numpy().T, self.duration_index)



