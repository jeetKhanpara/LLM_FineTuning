import json
import time
import os
import jsonlines
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# from templates import tags,template_1
from dataloader.templates import tags,template_1
from transformers import AutoTokenizer #(AutoTokenizer class from transformers library which belongs to hugging face)
from datasets import load_dataset
class DataLoader:
    """Data loader class"""

    def __init__(self) -> None:
        pass
    
    #for iterating the files present in interim ascending order
    @staticmethod
    def extract_number_from_filename(filename):
        start = filename.find("{")
        end = filename.find("}")

        return int(filename[start+1:end])
    
    #loading data from json file
    @staticmethod
    def load_data(path,start,end):
        """Loads the data from the data.json"""

        with open(path,"r") as json_file:
            data = json.load(json_file) #load fun will read whole json file and here data.json file contains list, so list of dictionary will be returned
        
        return data[start:end]        
    
    #data has been loaded in the form of an array 
    #we now want only discriptions from that array of dictionaries
    @staticmethod
    def process_data(data,start):
        processed_dataset = []

        for i,item in enumerate(data):
            data_to_append = {
                "id" : start+i,
                "description" : item["_source"]["description"]
            }
            processed_dataset.append(data_to_append)

        return processed_dataset

    #extract the tags from those description present in peocessed dataset,
    #we would have dataset with id,description and tags
    @staticmethod
    def extract_tags_from_description(processed_dataset):

        _ = os.getenv("OPENAI_API_KEY")
        gpt = ChatOpenAI(temperature=0.0)

        dataset_with_tags = []
        st = time.time()

        for index,datapoint in tqdm(enumerate(processed_dataset)):
            prompt_template = ChatPromptTemplate.from_template(template_1)
            message = prompt_template.format_messages(
                description=datapoint["description"],
                tags= tags
                )
            
            try:
                llm_response = gpt(message)
                predicted_tags = eval(llm_response.content) 
                #llm_response would be of array of string and that whole array itself is a string,so eval function will convert that string into perticular array
            except Exception as e:
                print("an error occured while getting response from model")

            new_dataset = {
                "id" : datapoint["id"],
                "description" : datapoint["description"],
                "tags" : predicted_tags
            }

            dataset_with_tags.append(new_dataset)
        et = time.time()
        time_taken = et - st
        print("time taken by llm to extract the tags is ",time_taken," seconds.")
        
        return dataset_with_tags

    #the extracted tags which are present in the array of dictionary, now it will be
    #stored in 
    @staticmethod
    def save_extracted_tags(dataset_with_tags,path):
        """Saves the dataset_with_tags to a json file"""

        try:
            with open(path,"w") as file:
                json.dump(dataset_with_tags,file,indent=2)
        except Exception as e:
                print("an error occured while saving into JSON file!",e)

        print("data stored in the json file successfully at path ",path)

    #updating the keys in each file present in interim folder and save it to dataset.jsonl
    @staticmethod
    def save_corrected_dataset(path):
        """takes the corrected json file and creates a final json"""

        final_corrected_dataset = []

        for filename in sorted(os.listdir(path),key=DataLoader.extract_number_from_filename):
            print(filename)
            with open(os.path.join(path,filename),"r") as file:
                interim_dataset = json.load(file)

                key_updated = []

                for datapoint in interim_dataset:
                    new_dict={
                        "id":datapoint["id"],
                        "input":datapoint["description"],
                        "output":datapoint["tags"] 
                    }
                    key_updated.append(new_dict)

            final_corrected_dataset.extend(key_updated)
        
        with jsonlines.open("../data/final/final_dataset.jsonl","w") as writer:
            writer.write_all(final_corrected_dataset)

    @staticmethod
    def tokenize_datapoint(finetuning_dataset_dictionary):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        description = finetuning_dataset_dictionary['input'][0]
        output = finetuning_dataset_dictionary['output'][0]
        text = description + str(output)

        tokenizer.pad_token = tokenizer.eos_token
        tokenized_inputs = tokenizer(
            text=text,
            return_tensors="np",
            padding=True
        )
        max_length = min(tokenized_inputs["input_ids"].shape[1],2048)
        tokenized_inputs = tokenizer(
            text=text,
            return_tensors="np",
            truncation=True,
            max_length=max_length
        )

        return tokenized_inputs
    
    @staticmethod
    def load_and_tokenize_final_dataset(path):
        finetuning_dataset_hf = load_dataset(
            "json",
            data_files=path,
            split="train"
        )

        tokenized_dataset = finetuning_dataset_hf.map(
            DataLoader.tokenize_datapoint,
            batched=True,
            batch_size=1,
            drop_last_batch=True
        )
        tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])
        splitted_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)

        return splitted_dataset["train"], splitted_dataset["test"]
    


        

    
# data = DataLoader.load_data("./data/raw/data.json",0,50)
# final_data = DataLoader.process_data(data,10)
# dataset_with_tags=DataLoader.extract_tags_from_description(final_data)
# DataLoader.save_extracted_tags(dataset_with_tags,"./data/interim/generated/data_{1}_to_{50}.json")
# DataLoader.save_corrected_dataset("data/interim/generated/")

