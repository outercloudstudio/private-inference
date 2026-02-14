from heir import compile
from heir.mlir import I64, Secret

@compile()  # defaults to scheme="bgv", OpenFHE backend, and debug=False
def func(a: Secret[I64], b: Secret[I64], c: Secret[I64], d: Secret[I64]):
    l_0_0 = 0 + a * 4 - b * 6 - c * 50 - d * 3 + 6
    l_0_1 = 0 - a * 7 + b * 12 - c * 33 + d * 5 + 7
    l_0_2 = 0 - a * 6 + b * 10 + c * 10 + d * 5 + 9
    l_0_3 = 0 + a * 2 - b * 4 + c * 29 - d * 2 + 19
    l_1_0 = 0 + l_0_0 * 21 + l_0_1 * 7 - l_0_2 * 5 - l_0_3 * 12 + 19
    l_1_1 = 0 + l_0_0 * 0 - l_0_1 * 6 + l_0_2 * 10 - l_0_3 * 4 - 31

    return l_1_0 - l_1_1

func.setup()

enc_a = func.encrypt_a(1)
enc_b = func.encrypt_b(-1)
enc_c = func.encrypt_c(-1)
enc_d = func.encrypt_d(1)

print("evaling...")

result_enc = func.eval(enc_a, enc_b, enc_c, enc_d)

print("decrypting...")

result = func.decrypt_result(result_enc)

print(
  f"Expected result for `func`: {func.original(1,-1, -1, 1)}, FHE result:"
  f" {result}"
)