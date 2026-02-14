from heir import compile
from heir.mlir import I64, Secret

@compile()  # defaults to scheme="bgv", OpenFHE backend, and debug=False
def func(a: Secret[I64], b: Secret[I64], c: Secret[I64], d: Secret[I64]):
    l_0_0 = 0 + a * 0 * 10  + b * 0 * 10  + c * 16 * 10  + d * 0 * 10  + 16
    l_0_0 = l_0_0 * l_0_0 / (50 * 50)
    l_0_1 = 0 + a * 0 * 10  + b * 0 * 10  - c * 15 * 10  + d * 0 * 10  - 15     
    l_0_1 = l_0_1 * l_0_1 / (50 * 50)
    l_0_2 = 0 + a * 0 * 10  + b * 0 * 10  - c * 16 * 10  + d * 0 * 10  + 15     
    l_0_2 = l_0_2 * l_0_2 / (50 * 50)
    l_0_3 = 0 - a * 1 * 10  + b * 0 * 10  - c * 1 * 10  + d * 0 * 10  - 1       
    l_0_3 = l_0_3 * l_0_3 / (50 * 50)
    l_1_0 = 0 - l_0_0 * 8 - l_0_1 * 7 + l_0_2 * 12 + l_0_3 * 0 + 1
    l_1_1 = 0 + l_0_0 * 8 + l_0_1 * 4 - l_0_2 * 10 + l_0_3 * 0 + 0

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