from heir import compile
from heir.mlir import I64, Secret

@compile()  # defaults to scheme="bgv", OpenFHE backend, and debug=False
def func(a: Secret[I64], b: Secret[I64], c: Secret[I64], d: Secret[I64]):
    output = a - b - c + d + 0

    return output

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