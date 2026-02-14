import json

# Read the JSON file
with open('balancing_weights.json', 'r') as f:
    model_params = json.load(f)

result = ""

for i in range(256):
    line =  f"l_0_{i} = 0.0"

    for j in range(196):
        weight = round(model_params['fc1']['weights'][i][j] * 100) / 100

        operation = "+"

        if weight < 0:
            operation = "-"
            weight = -weight

        line += " " + operation + " " + input_names[j] + " * " + str(weight)

    bias_operation = "+"

    bias = round(model_params['fc1']['bias'][i] * 100) / 100

    if bias < 0:
        bias_operation = "-"
        bias = -bias

    line += " " + bias_operation + " " + str(bias)

    result += line + "\n"
    result += f"l_0_{i} = l_0_{i} * l_0_{i}" + "\n"

# for i in range(4):
#     line =  f"l_0_{i} = 0.0"

#     for j in range(4):
#         weight = round(model_params['fc1']['weights'][i][j] * 100) / 100

#         operation = "+"

#         if weight < 0:
#             operation = "-"
#             weight = -weight

#         line += " " + operation + " " + input_names[j] + " * " + str(weight)

#     bias_operation = "+"

#     bias = round(model_params['fc1']['bias'][i] * 100) / 100

#     if bias < 0:
#         bias_operation = "-"
#         bias = -bias

#     line += " " + bias_operation + " " + str(bias)

#     result += line + "\n"
#     result += f"l_0_{i} = l_0_{i} * l_0_{i}" + "\n"

# for i in range(2):
#     line =  f"l_1_{i} = 0.0"

#     for j in range(4):
#         weight = round(model_params['fc2']['weights'][i][j] * 100) / 100

#         operation = "+"

#         if weight < 0:
#             operation = "-"
#             weight = -weight

#         line += " " + operation + " l_0_" + str(j) + " * " + str(weight)

#     bias_operation = "+"

#     bias = round(model_params['fc2']['bias'][i] * 100) / 100

#     if bias < 0:
#         bias_operation = "-"
#         bias = -bias

#     line += " " + bias_operation + " " + str(bias)

#     result += line + "\n"

#     with open('codegen.gen', 'w') as f:
#         json.dump(result, f)