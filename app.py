import os

def run_evaluation():
    os.system("python evaluate.py")

def run_training():
    os.system("python train.py")

def run_prediction():
    os.system("python predict.py")

def run_csv_prediction():
    os.system("python predict_csv.py")

def main():
    print("Welcome to the App!")
    print("Please select an option:")
    print("1. Evaluate")
    print("2. Train")
    print("3. Predict")
    print("4. Predict CSV")

    choice = input("Enter your choice (1-4): ")

    if choice == '1':
        run_evaluation()
    elif choice == '2':
        run_training()
    elif choice == '3':
        run_prediction()
    elif choice == '4':
        run_csv_prediction()
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
