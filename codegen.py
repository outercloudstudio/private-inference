import json

# Read the JSON file
with open('balancing_weights.json', 'r') as f:
    model_params = json.load(f)

input_names = ["a", "b", "c", "d"]

for i in range(4):
    line =  f"l_0_{i} = 0"

    for j in range(4):
        weight = model_params['fc1']['weights'][i][j]

        operation = "+"

        if weight < 0:
            operation = "-"
            weight = -weight

        line += " " + operation + " " + input_names[j] + " * " + str(weight) + " * 10 "

    bias_operation = "+"

    bias = model_params['fc1']['bias'][i]

    if bias < 0:
        bias_operation = "-"
        bias = -bias

    line += " " + bias_operation + " " + str(bias)

    print(line)

    print(f"l_0_{i} = l_0_{i} * l_0_{i} / (50 * 50)")

for i in range(2):
    line =  f"l_1_{i} = 0"

    for j in range(4):
        weight = model_params['fc2']['weights'][i][j]

        operation = "+"

        if weight < 0:
            operation = "-"
            weight = -weight

        line += " " + operation + " l_0_" + str(j) + " * " + str(weight)

    bias_operation = "+"

    bias = model_params['fc2']['bias'][i]

    if bias < 0:
        bias_operation = "-"
        bias = -bias

    line += " " + bias_operation + " " + str(bias)

    print(line)