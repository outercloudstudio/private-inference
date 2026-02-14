from heir import compile
from heir.mlir import I64, Secret

@compile()  # defaults to scheme="bgv", OpenFHE backend, and debug=False
def func(a: Secret[I64], b: Secret[I64], c: Secret[I64], d: Secret[I64]):
    l_0_0 = 0 + a * 0 * 10  + b * 0 * 10  + c * 2 * 10  + d * 0 * 10  + 3
    l_0_0 = l_0_0 * l_0_0
    l_0_1 = 0 - a * 1 * 10  + b * 1 * 10  - c * 1 * 10  + d * 0 * 10  + 0       
    l_0_1 = l_0_1 * l_0_1
    l_0_2 = 0 + a * 0 * 10  + b * 0 * 10  + c * 0 * 10  + d * 1 * 10  - 1       
    l_0_2 = l_0_2 * l_0_2
    l_0_3 = 0 + a * 0 * 10  - b * 1 * 10  + c * 0 * 10  + d * 0 * 10  - 1       
    l_0_3 = l_0_3 * l_0_3
    l_1_0 = 0 - l_0_0 * 1 + l_0_1 * 0 + l_0_2 * 0 - l_0_3 * 1 + 5
    l_1_1 = 0 + l_0_0 * 1 + l_0_1 * 0 + l_0_2 * 0 + l_0_3 * 0 - 4

    return l_1_0 - l_1_1

func.setup()

enc_a = func.encrypt_a(0)
enc_b = func.encrypt_b(0)
enc_c = func.encrypt_c(2)
enc_d = func.encrypt_d(0)

print("evaling...")

result_enc = func.eval(enc_a, enc_b, enc_c, enc_d)

print("decrypting...")

result = func.decrypt_result(result_enc)

print(
  f"Expected result for `func`: {func.original(0, 0, 2, 0)}, FHE result:"
  f" {result}"
)