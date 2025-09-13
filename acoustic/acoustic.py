from transformers import WavLMModel
import torch, peft, gputil

WAVLM_Model = WavLMModel.from_pretrained("microsoft/wavlm-base")

# Freeze wavlm
for i in WAVLM_Model.parameters():
    i.requires_grad = False

# LoRA
LORA_Config = peft.LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj"],
    bias="none",
    task_type=peft.TaskType.FEATURE_EXTRACTION,
    layers_to_transform=list(range(len(WAVLM_Model.encoder.layers) - 4, len(WAVLM_Model.encoder.layers))),
    layers_pattern="layers"
)

LORA_Model = peft.get_peft_model(WAVLM_Model, LORA_Config)
LORA_Model.print_trainable_parameters()
