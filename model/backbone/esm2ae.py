import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.esm.modeling_esm import EsmConfig, EsmModel, EsmPreTrainedModel

from ..utils.ActivationFunction import SwiGLU
from ..utils.Simclr import Simclr_loss


class ESM2AE(EsmPreTrainedModel):
    def __init__(self, config: EsmConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.esm = EsmModel(config)
        # self.whitening = Whitening(config.hidden_size)

        self.encoder = nn.Sequential(
            nn.Linear(config.hidden_size, 768),
            SwiGLU(),
            nn.RMSNorm(768),
            nn.Linear(768, 512),
            SwiGLU(),
            nn.RMSNorm(512),
            nn.Linear(512, 256),
            SwiGLU(),
            nn.RMSNorm(256),
            nn.Linear(256, 128),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            SwiGLU(),
            nn.RMSNorm(256),
            nn.Linear(256, 512),
            SwiGLU(),
            nn.RMSNorm(512),
            nn.Linear(512, 768),
            SwiGLU(),
            nn.RMSNorm(768),
            nn.Linear(768, config.hidden_size),
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_ids_2=None,
        attention_mask_2=None,
        simclr_weight: float = 0.1,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # === branch 1 ===
        out1 = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        seq1 = out1[0][:, 0, :]  # CLS token
        centered1 = seq1 - seq1.mean(dim=-1, keepdim=True)
        norm1 = F.normalize(centered1, p=2, dim=-1)
        z1 = self.encoder(norm1)

        # === decoder ===
        reconstructed = self.decoder(z1)
        reconstruction_loss = F.mse_loss(reconstructed, seq1.detach())

        # === optional SimCLR branch ===
        contrastive_loss = 0.0
        if input_ids_2 is not None:
            out2 = self.esm(
                input_ids=input_ids_2,
                attention_mask=attention_mask_2,
                return_dict=return_dict,
            )
            seq2 = out2[0][:, 0, :]
            centered2 = seq2 - seq2.mean(dim=-1, keepdim=True)
            norm2 = F.normalize(centered2, p=2, dim=-1)
            z2 = self.encoder(norm2)

            # SimCLR loss (InfoNCE)
            contrastive_loss = Simclr_loss(z1, z2)

        total_loss = reconstruction_loss + simclr_weight * contrastive_loss

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=reconstructed,
        )
