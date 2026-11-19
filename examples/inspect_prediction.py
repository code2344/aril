from aril.predictor import byte_cdf

prefix = b"Once upon a tim"
cdf = byte_cdf(prefix)

assert len(cdf) == 257
assert cdf[0] == 0 and cdf[-1] == 65_536
assert all(lo < hi for lo, hi in zip(cdf, cdf[1:]))

b = ord("e")
print(f"Next-byte interval for 'e': [{cdf[b]}, {cdf[b + 1]})")
print(f"Probability used by ARIL: {(cdf[b + 1] - cdf[b]) / cdf[-1]:.2%}")
