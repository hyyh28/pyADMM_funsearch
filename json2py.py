import json

# 读取JSON文件
def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 写入Python文件
def write_function_to_python(json_data, output_file):
    function_code = json_data.get("function", "")
    
    if function_code:
        with open(output_file, 'w') as f:
            f.write(function_code)
        print(f"Function code successfully written to {output_file}")
    else:
        print("No function found in JSON.")

# 主函数
def main():
    input_json = '/home/ubuntu/pyADMM_funsearch/logs/low_rank_matrix_models/sparsesc/funsearch_sparsesc/samples/samples_10.json'  # 输入的JSON文件路径
    output_py = 'output_function.py'  # 输出的Python文件路径

    # 读取JSON
    json_data = read_json(input_json)

    # 将function字段写入Python文件
    write_function_to_python(json_data, output_py)

if __name__ == "__main__":
    main()
