"""Half-up rounding (ties away from zero) for every number printed into the paper. The value is first
rounded to 9 decimals so that float noise (0.19499999... for an exact 0.195) does not decide a tie."""
from decimal import Decimal, ROUND_HALF_UP


def r2(x, nd=2):
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(Decimal("1e-9"), rounding=ROUND_HALF_UP).quantize(q, rounding=ROUND_HALF_UP))
