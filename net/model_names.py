"""Canonical model names for the VSSB/Mamba JSCC variant.

The original project used the ``SwinJSCC_*`` model names.  This fork replaces
Swin Transformer blocks with VSSBlock/Mamba blocks, so the canonical public
names are now ``VSSBJSCC_*``.  Legacy names are still accepted to avoid
breaking old commands, logs, and checkpoints.
"""

VSSBJSCC_WO_SA_RA = "VSSBJSCC_w/o_SAandRA"
VSSBJSCC_W_SA = "VSSBJSCC_w/_SA"
VSSBJSCC_W_RA = "VSSBJSCC_w/_RA"
VSSBJSCC_W_SA_RA = "VSSBJSCC_w/_SAandRA"

VSSB_MODEL_CHOICES = [
    VSSBJSCC_WO_SA_RA,
    VSSBJSCC_W_SA,
    VSSBJSCC_W_RA,
    VSSBJSCC_W_SA_RA,
]

LEGACY_TO_VSSB = {
    "SwinJSCC_w/o_SAandRA": VSSBJSCC_WO_SA_RA,
    "SwinJSCC_w/_SA": VSSBJSCC_W_SA,
    "SwinJSCC_w/_RA": VSSBJSCC_W_RA,
    "SwinJSCC_w/_SAandRA": VSSBJSCC_W_SA_RA,
}

MODEL_CHOICES = VSSB_MODEL_CHOICES + list(LEGACY_TO_VSSB.keys())


def normalize_model_name(model_name: str) -> str:
    """Return the canonical VSSBJSCC name for a current or legacy model name."""
    return LEGACY_TO_VSSB.get(model_name, model_name)


def uses_rate_adaptation(model_name: str) -> bool:
    model_name = normalize_model_name(model_name)
    return model_name in {VSSBJSCC_W_RA, VSSBJSCC_W_SA_RA}


def uses_snr_adaptation(model_name: str) -> bool:
    model_name = normalize_model_name(model_name)
    return model_name in {VSSBJSCC_W_SA, VSSBJSCC_W_SA_RA}
