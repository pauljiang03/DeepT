# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.
from datetime import datetime
from pathlib import Path
import time

import os
import torch
from typing import Tuple, Optional

from termcolor import colored

from Verifiers.Zonotope import cleanup_memory
from data_utils import sample, get_all_correctly_classified_samples, compute_accuracy

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

        if not args.with_lirpa_transformer:
            self.embeddings = target.model.bert.embeddings
            self.encoding_layers = target.model.bert.encoder.layer
            self.pooler = target.model.bert.pooler
            self.classifier = target.model.classifier
        else:  # LIRPA model
            self.embeddings = target.model.embeddings
            self.encoding_layers = target.model.model_from_embeddings.bert.encoder.layer
            self.pooler = target.model.model_from_embeddings.bert.pooler
            self.classifier = target.model.model_from_embeddings.classifier

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

        self.res_filename = "{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
            "resultsSynonym" if self.args.attack_type == "synonym" else "results",
            args.data if not self.args.one_word_per_sentence else args.data + "Subset",   # SST or YELP
            args.lirpa_ckpt.split("/")[1].split('_')[1] if self.args.attack_type == "synonym" else args.dir,  # 3, 6, 12
            size,
            "zonotopeSlow" if (args.method == "zonotope" and args.zonotope_slow) else args.method,  # zonotope vs baf
            args.p if args.p <= 10 else 'inf',  # 1, 2, inf
            "None_None" if reduc_method == "None" else f"{reduc_method}_{args.max_num_error_terms}",
            details,
            datetime.now().strftime('%b%d_%H-%M-%S')
        )
    
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
