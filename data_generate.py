from dataloader.dataloader import DataLoader
from save_final_jsonfile import FinalJson
from configs.config import CFG
import time

def main():
    dloader = DataLoader()
    start = CFG["data"]["startIndex"]
    end = CFG["data"]["endIndex"]+1
    final_path = CFG["data"]["path"]

    dataset = dloader.load_data(final_path,start,end)
    processed_dataset = dloader.process_data(dataset,CFG["data"]["startIndex"])
    dataset_with_tags = dloader.extract_tags_from_description(processed_dataset)
    dloader.save_extracted_tags(dataset_with_tags,"./data/interim/generated/data_{101}_to_{150}.json")
    
    
# DataLoader.save_corrected_dataset("data/interim/generated/")

    # with open("final_dataset.txt",'w') as file:
    #         file.write("")
    
    # for i in range(len(dataset_with_tags)):    
    #     with open("final_dataset.txt","a") as file:
    #         file.write(str(dataset_with_tags[i]))


if __name__=="__main__":
    main()