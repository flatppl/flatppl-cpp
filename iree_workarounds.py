"""Prepare FlatPPL's textual StableHLO for the stock IREE compiler."""

import re

# This matches the single-line form emitted by FlatPPL, including its precision.
_DYNAMIC_MATVEC = re.compile(
    r"^(?P<result>[ \t]*%[\w.$-]+ = )stablehlo\.dot_general "
    r"(?P<operands>%[\w.$-]+, %[\w.$-]+), contracting_dims = \[1\] x \[0\], "
    r"(?P<signature>precision = \[(?:DEFAULT|HIGH|HIGHEST), (?:DEFAULT|HIGH|HIGHEST)\] : "
    r"\(tensor<\?x(?P<width>[0-9]+)x(?P<dtype>f32|f64)>, tensor<(?P=width)x(?P=dtype)>\)"
    r" -> tensor<\?x(?P=dtype)>)$",
    re.MULTILINE,
)


def apply_iree_workarounds(stablehlo: str) -> str:
    # IREE's general contraction lowering introduces an invalid dynamic reshape.
    return _DYNAMIC_MATVEC.sub(
        r"\g<result>stablehlo.dot \g<operands>, \g<signature>", stablehlo
    )
