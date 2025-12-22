# defense_hparams.py

def register_hparams(defense: type):
    hparams = {}

    def _hparam(name, default_value):
        assert name not in hparams
        hparams[name] = default_value

    if defense.__name__ == "SmoothLLM":
        _hparam("number_of_copies", 1)
        _hparam("perturbation_type", "RandomSwapPerturbation")
        _hparam("perturbation_pct", 5)

    else:
        raise ValueError(f"Defense {defense.__name__} not found in defense_hparams.py.")

    return hparams