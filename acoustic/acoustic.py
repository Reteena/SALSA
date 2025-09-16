from transformers import WavLMModel
import torch, peft

class AttentionPooling(torch.nn.Module):
    def __init__(self, hidden_size=768, heads=12):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        pass

class AcousticBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        WAVLM_Model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.wavlm = WAVLM_Model # Definition here

        # Freeze wavlm
        for i in WAVLM_Model.parameters():
            i.requires_grad = False

        # LoRA
        LORA_Config = peft.LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none",
            task_type=peft.TaskType.FEATURE_EXTRACTION,
            layers_to_transform=list(range(len(WAVLM_Model.encoder.layers) - 4, len(WAVLM_Model.encoder.layers))),
            layers_pattern="layers"
        )

        self.wavlm = peft.get_peft_model(WAVLM_Model, LORA_Config)
        self.attention_pooling = AttentionPooling(self.wavlm.config.hidden_size)

    def forward(self,x):
        x = self.wavlm(x)
        x = self.attention_pooling(x)
        return x

class AcousticProbe(torch.nn.Module):
    def __init__(self, hidden, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.linear(x)
