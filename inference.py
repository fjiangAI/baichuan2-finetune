import json
import torch
import argparse
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from accelerate import Accelerator

# Custom Dataset Class for handling the test data
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_length=1536):
        # Load dataset from the specified path
        with open(data_path, encoding='utf-8') as f:
            self.dataset = json.load(f)

        # Initialize tokenizer and tokens
        self.tokenizer = tokenizer
        self.user_token = self.tokenizer.convert_ids_to_tokens(195)
        self.assistant_token = self.tokenizer.convert_ids_to_tokens(196)
        self.max_length = max_length

        # Prepare data samples
        self.datas = [
            {
                'input_token': self.generate_prompt(sample["conversations"])[0],
                "reference": self.generate_prompt(sample["conversations"])[1]
            }
            for sample in self.dataset
        ]

    def generate_prompt(self, conversation):
        # Generate prompt tokens based on the conversation
        input_token = ""
        for message in conversation[:-1]:
            token = self.user_token if message["from"] == "human" else self.assistant_token
            input_token += token + message["value"]
        input_token += self.assistant_token

        # The last message is considered as the reference response
        reference_token = conversation[-1]["value"]
        return input_token, reference_token

    def __getitem__(self, index):
        # Fetch a single data sample
        data = self.datas[index]
        return {
            'reference': data['reference'],
            'input': data['input_token']
        }

    def __len__(self):
        # Return the total number of samples
        return len(self.datas)

    def collate_fn(self, batch):
        # Custom collate function for DataLoader
        batch_input = [x['input'] for x in batch]
        batch_reference = [x['reference'] for x in batch]

        # Tokenize the inputs
        output_tokenizer = self.tokenizer(batch_input, return_tensors='pt', padding='longest')
        input_ids = output_tokenizer['input_ids']
        attention_mask = output_tokenizer['attention_mask']

        # Truncate inputs to max_length if necessary
        if input_ids.shape[-1] > self.max_length:
            input_ids = input_ids[:, -self.max_length:]
            attention_mask = attention_mask[:, -self.max_length:]

        return {
            'reference': batch_reference,
            'input': batch_input,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# Function to decode and gather responses from model outputs
def get_response(inputs, outputs, tokenizer, num_return=1):
    responses_list = []
    batch_return = []
    for i, output in enumerate(outputs):
        input_len = len(inputs[0])
        generated_output = output[input_len:]
        batch_return.append(tokenizer.decode(generated_output, skip_special_tokens=True))
        if i % num_return == num_return - 1:
            responses_list.append(batch_return)
            batch_return = []
    return responses_list

# Main testing function
def test(args):
    accelerator = Accelerator()
    torch.cuda.set_device(accelerator.process_index)

    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.cuda().eval()

    # Load and prepare the dataset and dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='right')
    tokenizer.pad_token_id = 0
    dataset = TestDataset(args.data_path, tokenizer)
    val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)
    val_dataloader = accelerator.prepare(val_dataloader)

    # Generation arguments
    gen_kwargs = {'num_return_sequences': args.num_return_sequences, 'max_new_tokens': args.max_new_tokens}

    # Initialize caches for storing results
    cache_reference, cache_response, cache_input = [], [], []

    # Start the testing loop
    with torch.no_grad():
        dataloader_iterator = tqdm(val_dataloader, total=len(val_dataloader)) if accelerator.is_main_process else val_dataloader
        for batch in dataloader_iterator:
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
            response = get_response(input_ids, outputs, tokenizer, args.num_return_sequences)

            cache_reference.extend(batch["reference"])
            cache_response.extend(response)
            cache_input.extend(batch["input"])

        # Gather results from all processes
        all_reference, all_response, all_input = gather_results(cache_reference, cache_response, cache_input)

        # Save the test results
        if accelerator.is_main_process:
            save_results(args.out_file, all_reference, all_response, all_input)

# Function to gather results from all distributed processes
def gather_results(cache_reference, cache_response, cache_input):
    all_reference = [None] * dist.get_world_size()
    all_response = [None] * dist.get_world_size()
    all_input = [None] * dist.get_world_size()

    dist.all_gather_object(all_response, cache_response)
    dist.all_gather_object(all_reference, cache_reference)
    dist.all_gather_object(all_input, cache_input)

    # Flatten the lists
    all_reference = [item for sublist in all_reference for item in sublist]
    all_response = [item for sublist in all_response for item in sublist]
    all_input = [item for sublist in all_input for item in sublist]
    return all_reference, all_response, all_input

# Function to save the test results to a file
def save_results(out_file, all_reference, all_response, all_input):
    ress = [
        {"reference": ref, "response": resp, "input": inp}
        for ref, resp, inp in zip(all_reference, all_response, all_input)
    ]
    print(f'test results: {out_file}')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(ress, f, ensure_ascii=False, indent=2)

# Main script execution: Argument parsing and initiating the test
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for testing the model')
    parser.add_argument('--data_path', default='./data/test.json', type=str)
    parser.add_argument('--model_path', default='./model_path/', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--num_return_sequences', default=1, type=int)
    parser.add_argument('--out_file', default='./output.json', type=str)
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Start the testing process
    test(args)



