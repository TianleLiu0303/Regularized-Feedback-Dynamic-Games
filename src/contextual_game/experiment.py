"""Experiment variants and separate result directories."""
from dataclasses import replace
from pathlib import Path


VARIANTS = {
    "full": ("demonstration", "adaptive"),
    "fixed": ("demonstration", "fixed"),
    "cost-reference": ("cost", "adaptive"),
    "no-kl": ("none", "adaptive"),
}


def configure_variant(config, method, variant=None):
    """Reference ablations reuse the contextual checkpoint, not the KLGame MLP."""
    if method == "klgame":
        if variant is not None:
            raise ValueError("--variant uses --method main and its contextual checkpoint")
        if (config.reference_source, config.coefficient_mode) != ("cost", "fixed"):
            raise ValueError("KLGame requires a cost reference and fixed coefficient")
        return config
    if variant is not None:
        source, mode = VARIANTS[variant]
        return replace(config, reference_source=source, coefficient_mode=mode)
    if (config.reference_source, config.coefficient_mode) not in VARIANTS.values():
        raise ValueError("unsupported contextual reference/coefficient combination")
    return config


def variant_name(config):
    return next(name for name, pair in VARIANTS.items()
                if pair == (config.reference_source, config.coefficient_mode))


def prepare_results(directory):
    """Reject old or partial runs before writing any new results."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("result directory is not empty; choose a new --output for this run")
    return directory
