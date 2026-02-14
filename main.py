from heir import compile
from heir.mlir import I64, Secret

@compile()  # defaults to scheme="bgv", OpenFHE backend, and debug=False
def func(a: Secret[I64], b: Secret[I64], c: Secret[I64], d: Secret[I64]):
    l_0_0 = 0 + a * 21 + b * 9 - c * 37 - d * 9 - 4
    l_0_1 = 0 - a * 10 + b * 0 - c * 60 + d * 1 + 4
    l_0_2 = 0 - a * 16 - b * 1 + c * 22 + d * 2 + 6
    l_0_3 = 0 - a * 27 - b * 12 - c * 15 + d * 11 + 5
    l_1_0 = 0 + l_0_0 * 11 + l_0_1 * 10 - l_0_2 * 8 + l_0_3 * 3 + 21
    l_1_1 = 0 - l_0_0 * 8 - l_0_1 * 6 - l_0_2 * 4 - l_0_3 * 8 + 20

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