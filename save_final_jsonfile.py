from dataloader.dataloader import DataLoader
#saving all the data togethere present in interim folder
#these all data will be saved in dataset.jsonl
def main():
 
    floader = DataLoader()
    path="data/interim/generated"

    floader.save_corrected_dataset(path)

if __name__=="__main__":
    main()

# class FinalJson(DataLoader):

#     def __init__(self) -> None:
#         super().__init__()
#         pass


#     @staticmethod
#     def save_corrected_dataset(path):
#         """takes the corrected json file and creates a final json"""

#         final_corrected_dataset = []

#         for filename in sorted(os.listdir(path),key=DataLoader.extract_number_from_filename):
#             print(filename)
#             with open(os.path.join(path,filename),"r") as file:
#                 interim_dataset = json.load(file)

#                 key_updated = []

#                 for datapoint in interim_dataset:
#                     new_dict={
#                         "id":datapoint["id"],
#                         "input":datapoint["description"],
#                         "output":datapoint["tags"] 
#                     }
#                     key_updated.append(new_dict)

#             final_corrected_dataset.extend(key_updated)
        
#         with jsonlines.open("dataset.jsonl","w") as writer:
#             writer.write_all(final_corrected_dataset)