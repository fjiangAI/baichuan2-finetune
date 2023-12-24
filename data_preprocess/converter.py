import json
import argparse

class Converter:
    def __init__(self) -> None:
        pass
                        
    def process_test_file(self, input_file_path, output_file_path):
        """读取文件，处理数据，然后保存到新文件。"""
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        data = self.accumulate_conversations(data)
        print(len(data))
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    # Adjusting the function to accumulate conversations as described
    def accumulate_conversations(slef, data):
        new_data = []
        for item in data:
            conversations = item['conversations']
            num_pairs = len(conversations) // 2  # Assuming each conversation consists of a pair of human and gpt
            
            for i in range(1, num_pairs + 1):
                end_index = i * 2  # Each conversation has 2 entries (human and gpt)
                new_conversations = conversations[:end_index]
                new_data.append({
                    "id": item["id"],
                    "conversations": new_conversations
                })
        return new_data

# Apply the function to the original data
        
def main():
    parser = argparse.ArgumentParser(description="Process and convert conversation data.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file.")
    parser.add_argument("--output", required=True, help="Path to the output JSON file.")
    
    args = parser.parse_args()

    # 实例化 Converter 并处理文件
    converter = Converter()
    converter.process_test_file(args.input, args.output)
if __name__ == "__main__":
    main()


            