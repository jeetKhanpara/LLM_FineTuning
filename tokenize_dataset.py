path = "./dataset.jsonl"
from dataloader.dataloader import DataLoader

def main():
    the_final_dataset = DataLoader.load_and_tokenize_final_dataset(path)
    print(type(the_final_dataset))
    with open("./data/final/train_test_data.txt","w") as file:
        file.write(str(the_final_dataset))

        
if __name__ == "__main__":
    main()