import json
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional


@dataclass
class FreezeArguments:
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    """

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_extra_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_module_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": 'Prefix of frozen modules for partial-parameter (freeze) fine-tuning. \
                  Use commas to separate multiple modules.\
                  LLaVA-HF choices: ["vision_tower"], \
                  YI-VL choices: ["model.vision_tower"], \
                  Qwen-VL choices: ["transformer.visual.transformer", "transformer.visual.conv1"]'
        },
    )


@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    use_rslora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."}
    )
    create_new_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class RLHFArguments:
    r"""
    Arguments pertaining to the PPO and KTO training.
    """

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in preference training."},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo", "ift"] = field(
        default="sigmoid",
        metadata={"help": "The type of Preference Optimization (PO) loss to use."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust label smoothing parameter that should be between 0 and 0.5."},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training."},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ift_gamma: float = field(
        default=0.95,
        metadata={"help": "The target reward margin term in IFT loss."},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step."},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."},
    )
    ref_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO training."},
    )
    ref_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    ref_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    reward_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    reward_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )

@dataclass
class ReftArguments:
    reft_layers: str = field(
        default="all",
        metadata={
            "help": (
                "which layers to intervene on"
                "Use commas to separate layer index, eg: 1,2,3"
                "Use `all` to specify all the available layers."
            )
        },
    )
    reft_rank: Optional[int] = field(
        default=4,
        metadata={"help": "The intrinsic dimension for ReFT tuning."},
    )
    reft_intervention_type: Literal['NoreftIntervention', 'LoreftIntervention',
                                    'ConsreftIntervention','LobireftIntervention',
                                    'DireftIntervention', 'NodireftIntervention'] = field(
        default="LoreftIntervention",
        metadata={"help": "The type of Reft intervention"},
    )

@dataclass
class GRPOArguments:
    r"""
    Arguments pertaining to the GRPO (Group Relative Policy Optimization) training.
    """

    grpo_num_generations: int = field(
        default=8,
        metadata={"help": "Number of generations per prompt for GRPO training. Must be at least 2."}
    )
    grpo_num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."}
    )
    grpo_format_weight: float = field(
        default=0.3,
        metadata={"help": "Weight for format reward (checking <think> and <answer> tags)."}
    )
    grpo_accuracy_weight: float = field(
        default=0.7,
        metadata={"help": "Weight for accuracy reward (answer correctness)."}
    )
    grpo_fps_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for FPS matching reward (comparing predicted fps with ground truth)."}
    )
    grpo_time_iou_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for time span IoU reward (comparing predicted span with ground truth)."}
    )
    grpo_loss_type: Literal["grpo", "dapo", "dr_grpo", "bnpo", "cispo", "sapo"] = field(
        default="dapo",
        metadata={
            "help": (
                "GRPO loss type. 'grpo': standard (not recommended due to length bias), "
                "'dapo': eliminates length bias (recommended), 'dr_grpo': global normalization, "
                "'bnpo': batch-normalized, 'cispo': clipped importance sampling, "
                "'sapo': soft adaptive policy optimization."
            )
        }
    )
    grpo_beta: float = field(
        default=0.0,
        metadata={"help": "KL divergence coefficient. 0.0 means no reference model (faster training)."}
    )
    grpo_epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping in GRPO loss."}
    )
    grpo_scale_rewards: Literal["group", "batch", "none"] = field(
        default="group",
        metadata={
            "help": (
                "Reward scaling strategy. 'group': scale by standard deviation within groups, "
                "'batch': scale across entire batch, 'none': no scaling."
            )
        }
    )
    grpo_mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "Exclude truncated completions from loss calculation to prevent noise during training."
        }
    )
    multi_objective_aggregation: Literal["sum_then_normalize", "normalize_then_sum"] = field(
        default="sum_then_normalize",
        metadata={
            "help": (
                "Method to aggregate multiple reward functions. "
                "'sum_then_normalize': First sums weighted rewards, then normalizes (default). "
                "'normalize_then_sum': First normalizes each reward function within groups, "
                "then sums using weights. Recommended by GDPO paper for multi-reward RL."
            )
        }
    )

@dataclass
class DistillationArguments:
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    """

    distill_loss_weight: float = field(
        default=0.5,
        metadata={
            "help": (
                "Distillation loss ratio versus soft target loss ratio in distillation training"
                "SFT loss ratio is 1 - distill_loss_weight"
            )
        },
    )
    kd_temperature: float = field(
        default=1,
        metadata={
            "help": (
                "kd_temperature"
            )
        },
    )
    teacher_temperature: float = field(
        default=1,
        metadata={
            "help": (
                "teacher_temperature"
            )
        },
    )
    kd_objective: Optional[Literal["forward_kl", "reverse_kl", "adaptive_kl", "skewed_forward_kl", "skewed_reverse_kl", "js_divergence"]] = field(
        default="forward_kl",
        metadata={
            "help": 
                ("kd_objective.")
        },
    )
    adaptive_kl_alpha: Optional[float] = field(
        default=0.5,
        metadata={
            "help": 
                ("adaptive_kl_alpha.")
        },
    )
    skew_lambda: Optional[float] = field(
        default=0.1,
        metadata={
            "help": 
                ("skew_lambda.")
        },
    )
    distill_on_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to distill on the prompt or not."},
    )

@dataclass
class FinetuningArguments(FreezeArguments, LoraArguments, RLHFArguments, ReftArguments, GRPOArguments, DistillationArguments):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Optional[Literal["sft", "grpo"]] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Optional[Literal["lora", "freeze", "full", "reft"]] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    disable_version_checking: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable version checking."},
    )
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={"help": "The number of eval times to wait before early stopping."},
    )
    early_stopping_threshold: Optional[float] = field(
        default=0.0,
        metadata={"help": "The threshold to use for early stopping."},
    )
    lr_scheduler_kwargs_json: Optional[str] = field(
        default=None,
        metadata={
            "help": "The JSON string of the keyword arguments for the learning rate scheduler. "
            "e.g. --lr_scheduler_kwargs_json='{\"num_cycles\": 0.3}'"
        },
    )
    use_turbo: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use the transformers_turbo training."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: List[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_module_prefix: Optional[List[str]] = split_arg(self.freeze_module_prefix)
        self.freeze_extra_modules: Optional[List[str]] = split_arg(self.freeze_extra_modules)
        self.lora_alpha = self.lora_alpha or self.lora_rank * 2
        if self.lora_target and not any(c in self.lora_target for c in ["*", "$", "|", "("]):
            # split when lora_target is not regex expression
            self.lora_target = split_arg(self.lora_target)
        self.additional_target: Optional[List[str]] = split_arg(self.additional_target)

        self.use_ref_model = False  # DPO removed

        assert self.finetuning_type in ["lora", "freeze", "full", "reft"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.lr_scheduler_kwargs_json is not None:
            try:
                self.lr_scheduler_kwargs_json = json.loads(self.lr_scheduler_kwargs_json)
            except json.JSONDecodeError:
                raise ValueError(
                    f"`lr_scheduler_kwargs_json` {self.lr_scheduler_kwargs_json} is not a valid JSON string."
                )


        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for the Freeze or LoRA method.")


        if self.finetuning_type == 'reft' and not self.reft_layers:
            raise ValueError("`reft_layers` is necessary for reft tuning.")

    def save_to_json(self, json_path: str):
        r"""Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        r"""Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()

        return cls(**json.loads(text))
