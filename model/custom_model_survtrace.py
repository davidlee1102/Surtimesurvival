from SurvTRACE.survtrace.modeling_bert import BertEmbeddings

from SurvTRACE.survtrace.utils import set_random_seed
from SurvTRACE.survtrace.model import SurvTraceSingle
from SurvTRACE.survtrace.config import STConfig
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
    ):
        output = self.our_model(input_nums, our_mask)
        print(output.shape)
        x = super().forward(input_ids=output[:, :0].long(), input_nums=output)
        return x
