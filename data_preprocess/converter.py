import json
import argparse

class Converter:
    def __init__(self) -> None:
        # Initialize the Converter class.
        pass
                        
    def process_test_file(self, input_file_path, output_file_path):
        """Read the file, process the data, and then save it to a new file."""
        # Open and read the input JSON file.
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Process the data to accumulate conversations.
        data = self.accumulate_conversations(data)
        print(len(data))
        
        # Write the processed data to the output JSON file.
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    
    def accumulate_conversations(self, data):
        """Reformat the data to accumulate conversations."""
        new_data = []
        # Iterate over each item in the data.
        for item in data:
            conversations = item['conversations']
            # Calculate the number of conversation pairs.
            num_pairs = len(conversations) // 2  # Assuming each conversation consists of a pair of human and GPT
            
            # Accumulate conversations up to the current pair.
            for i in range(1, num_pairs + 1):
                end_index = i * 2  # Each conversation has 2 entries (human and GPT)
                new_conversations = conversations[:end_index]
                # Append the accumulated conversation to the new data.
                new_data.append({
                    "id": item["id"],
                    "conversations": new_conversations
                })
        return new_data

def main():
    # Set up argument parsing for command-line functionality.
    parser = argparse.ArgumentParser(description="Process and convert conversation data.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file.")
    parser.add_argument("--output", required=True, help="Path to the output JSON file.")
    
    args = parser.parse_args()

    # Instantiate a Converter and process the file.
    converter = Converter()
    converter.process_test_file(args.input, args.output)

# Ensure that the script is being run directly (not imported) before executing.
if __name__ == "__main__":
    main()
