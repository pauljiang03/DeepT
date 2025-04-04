import os
import torch
from typing import Tuple, Optional

class Verifier:
    def __init__(self, args, target, logger):
        self.args = args
        self.device = args.device
        self.target = target
        self.logger = logger
        self.res = args.res
        self.results_directory = args.results_directory

        self.p = args.p if args.p < 10 else float("inf")
        self.eps = args.eps
        self.debug = args.debug
        self.verbose = args.debug or args.verbose
        self.method = args.method
        self.num_verify_iters = args.num_verify_iters
        self.max_eps = args.max_eps
        self.debug_pos = args.debug_pos
        self.perturbed_words = args.perturbed_words
        self.warmed = False

        self.hidden_act = args.hidden_act
        self.layer_norm = target.model.config.layer_norm if hasattr(target.model.config, "layer_norm") else "standard"

        reduc_method = args.error_reduction_method

        if "smaller" in args.dir:
            size = "smaller"
        elif "small" in args.dir:
            size = "small"
        else:
            size = "big"

        details = "NoConstraint" if (args.method != "zonotope" or not args.add_softmax_sum_constraint) else "WithConstraint"
        if self.args.use_other_dot_product_ordering:
            details += "OtherDotProductOrder"
        if self.args.use_dot_product_variant3:
            details += "DotProduceVariant3"
        if self.args.num_fast_dot_product_layers_due_to_switch != -1:
            if self.args.variant2plus1:
                details += f"Variant21With{self.args.num_fast_dot_product_layers_due_to_switch}FastLayers"
            else:
                details += f"Variant12With{self.args.num_fast_dot_product_layers_due_to_switch}FastLayers"
            details += f"WithNoise{self.args.max_num_error_terms_fast_layers}Symbols"


    
    def run(self, data):
        pass
    
    def start_verification_new_input(self):
        pass

    def reset(self):
        pass

    def get_bounds_difference_in_scores(
            self, embeddings: torch.Tensor, index: int, eps: float, gradient_enabled=False, *args, **kwargs
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        pass

    def verify(self, example, example_num: int, first_example=False):
        pass

    def verify_safety(self, example, embeddings, index, eps, **kwargs) -> bool:
        pass
