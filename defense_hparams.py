# defense_hparams.py

def register_hparams(defense: type):
    hparams = {}

    def _hparam(name, default_value):
        assert name not in hparams
        hparams[name] = default_value

    if defense.__name__ == "SmoothLLM":
        _hparam("number_of_copies", 2)
        _hparam("perturbation_type", "RandomSwapPerturbation")
        _hparam("perturbation_pct", 10)

    elif defense.__name__ == "PerplexityFilter":
        _hparam("perplexity_model_path", "gpt2")

    elif defense.__name__ == "EraseAndCheck":
        _hparam("erase_length", 1)
        _hparam("tokenizer_path", "gpt2")

    else:
        raise ValueError(f"Defense {defense.__name__} not found in defense_hparams.py.")

    return hparams