from heir import compile
from heir.mlir import F64, Secret

@compile(
    scheme="ckks",
    config={
        "logN": 14,
        "Q": [60, 40, 40, 40, 60],
        "P": [60, 60],
        "logDefaultScale": 40,
    }
)
def func(a: Secret[F64], b: Secret[F64], c: Secret[F64], d: Secret[F64]):
    l_0_0 = 0.0 + a * 0.04 + b * 0.04 - c * 6.24 - d * 0.04 + 2.07
    l_0_0 = l_0_0 * l_0_0
    l_0_1 = 0.0 + a * 0.09 + b * 0.05 + c * 7.41 - d * 0.05 + 1.96
    l_0_1 = l_0_1 * l_0_1
    l_0_2 = 0.0 + a * 0.22 + b * 0.07 + c * 4.74 - d * 0.15 - 1.32
    l_0_2 = l_0_2 * l_0_2
    l_0_3 = 0.0 - a * 0.1 - b * 0.06 + c * 6.19 + d * 0.09 + 1.63
    l_0_3 = l_0_3 * l_0_3
    l_1_0 = 0.0 + l_0_0 * 2.43 - l_0_1 * 1.71 + l_0_2 * 0.64 - l_0_3 * 1.34 - 0.2
    l_1_1 = 0.0 - l_0_0 * 1.93 + l_0_1 * 2.03 - l_0_2 * 0.94 + l_0_3 * 1.1 - 0.13

    return l_1_0 - l_1_1

func.setup()

enc_a = func.encrypt_a(0.0)
enc_b = func.encrypt_b(0.0)
enc_c = func.encrypt_c(2.0)
enc_d = func.encrypt_d(0.0)

print("evaling...")

result_enc = func.eval(enc_a, enc_b, enc_c, enc_d)

print("decrypting...")

result = func.decrypt_result(result_enc)

print(
  f"Expected result for `func`: {func.original(0.0, 0.0, 2.0, 0.0)}, FHE result:"
  f" {result}"
)