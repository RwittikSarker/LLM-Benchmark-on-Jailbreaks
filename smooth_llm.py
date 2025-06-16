import copy
import random
from typing import Tuple
import numpy as np
import pandas as pd
from perturbations import (
    RandomSwapPerturbation,
    RandomPatchPerturbation,
    RandomInsertPerturbation,
)
from defense_hparams import register_hparams
from judge_func import judge_df


class SmoothLLM:
    def __init__(self, target_model):
        self.target_model = target_model
        self.hparams = register_hparams(type(self))

        perturb_type = self.hparams["perturbation_type"]
        perturb_pct = self.hparams["perturbation_pct"]

        if perturb_type == "RandomSwapPerturbation":
            self.perturbation_fn = RandomSwapPerturbation(q=perturb_pct)
        elif perturb_type == "RandomInsertPerturbation":
            self.perturbation_fn = RandomInsertPerturbation(q=perturb_pct)
        elif perturb_type == "RandomPatchPerturbation":
            self.perturbation_fn = RandomPatchPerturbation(q=perturb_pct)
        else:
            raise ValueError(f"Invalid perturbation type: {perturb_type}")

    def query(self, prompt: list[dict[str, str]]) -> Tuple[str, bool]:
        perturbed_prompts = self.perturb_copies(prompt)
        responses = self.target_model.query_llm(perturbed_prompts)

        user_prompts = [p[1]["content"] for p in perturbed_prompts]
        model_responses = responses.responses if hasattr(responses, 'responses') else responses

        df_results = pd.DataFrame({
            "prompt": user_prompts,
            "response": model_responses
            })

        are_copies_jailbroken = judge_df(df_results, task="jailbreak")

        jb_percentage = np.mean(are_copies_jailbroken)
        is_smoothllm_jailbroken = jb_percentage > 0.5

        valid_choices = [
            r for r, jb in zip(model_responses, are_copies_jailbroken)
            if jb == is_smoothllm_jailbroken
        ]

        if valid_choices:
            final_response = random.choice(valid_choices)
        else:
            final_response = random.choice(model_responses) if model_responses else ""

        return final_response, is_smoothllm_jailbroken

    def perturb_copies(self, prompt):
        perturbed_prompts = []
        for _ in range(self.hparams["number_of_copies"]):
            prompt_copy = copy.deepcopy(prompt)
            prompt_copy[1]["content"] = self.perturbation_fn(prompt_copy[1]["content"])
            perturbed_prompts.append(prompt_copy)
        return perturbed_prompts