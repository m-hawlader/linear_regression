import csv
import random
import matplotlib.pyplot as plt

w_mod = 1
b_mod = 1
batches = 10000
data_PATH = r"D:\Path_to_ML\Projects\ReLU\age_net-worth.csv"
behaviour_PATH = PATH = r"D:\Path_to_ML\Projects\ReLU\behaviour.csv"

def get_behaviour():
    global data_b
    file = open(behaviour_PATH,"r", encoding="utf-8-sig")
    reader = csv.reader(file)
    data_b = []
    for lines in reader:
        data_b.append(lines)
    data_b = data_b[0]
    data_b[0] = float(data_b[0])
    data_b[1] = float(data_b[1])
    file.close()

def get_data():
    global data
    file = open(data_PATH,"r", encoding="utf-8-sig")
    lines = csv.reader(file)
    data = []
    for line in lines:
        data.append([float(line[0]), float(line[1])])
    file.close()

def log_behaviour(w,b):
    global data_b
    data_b = [w,b]
    file = open(behaviour_PATH,"w", encoding="utf-8-sig")
    writer = csv.writer(file)
    writer.writerow([w, b])
    file.close()

class model:
    def __init__(self):
        self.weight = round(data_b[0] + random.uniform(-1,1)*w_mod, 7)
        self.bias = round(data_b[1] + random.uniform(-1,1)*b_mod, 7)

    def predict(self,n):
        estimate = n * self.weight + self.bias
        if estimate < 0:
            estimate = 0
        return estimate
     
    def trial(self):
        error = 0
        for record in data:
            miss = (record[1] - self.predict(record[0]))**2
            error += miss
        return error

def train():
    models = []
    errors = []
    for i in range(batches):
        models.append(model())
        errors.append(models[i].trial())
    idx = errors.index(min(errors))
    log_behaviour(models[idx].weight, models[idx].bias)


def true_predict(n):
    estimate = data_b[0]*n + data_b[1]
    if estimate < 0:
        estimate = 0
    return estimate

def graph():
    get_behaviour()
    get_data()
    
    # Extract the actual data
    ages = [record[0] for record in data]
    net_worths = [record[1] for record in data]
    
    # Generate model predictions for the same ages
    predictions = [true_predict(age) for age in ages]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(ages, net_worths, label='Actual Data', color='blue', alpha=0.5)
    plt.plot(ages, predictions, label='Model Prediction', color='red', linewidth=2)
    plt.title('Age vs. Net Worth Prediction')
    plt.xlabel('Age')
    plt.ylabel('Net Worth (£)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    global w_mod
    global b_mod
    global batches
    get_behaviour()
    get_data()
    boolean = {
        "y": True,
        "n": False,
        "admin": "admin"
    }
    while True:
        choice = input("Do you want to train the model? (y/n/admin):\t").strip().lower()
        if choice not in boolean:
            print("Invalid option! Type 'y', 'n', or 'admin'.\n")
            continue
        wanna_train = boolean[choice]
        if wanna_train == True:
            n = int(input("How many time do you want to train the AI model:\t"))
            for i in range(n):
                train()
                print(f"Pass {i+1}")
        elif wanna_train == "admin":
            print("\n")
            w_mod = float(input("Enter weight modifier:\t"))
            b_mod = float(input("Enter bias modifier:\t"))
            batches = int(input("Enter batch size:\t"))
        n = int(input("\nThe model can predict net-worth:\nEnter your age:\t"))
        print(f"Your net-worth is £{true_predict(n):.2f}")
        cont = input("\nDo you want to continue? (y/n):\t").strip().lower()
        if cont != 'y':
            print("Exiting... Goodbye!")
            break


if __name__ == "__main__":
    main()
    graph()