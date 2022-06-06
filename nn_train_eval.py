

from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt

# reading data
def read_data(file_name):
    y_pred = []
    y_true = []
    with open(file_name, encoding="utf-8") as file:
            for line in file.readlines():
                y_pred.append(line.split(",")[0])
                y_true.append(line.split(",")[1][:-1])
    return y_pred, y_true

# saving new values
def new_metrics():
    y_pred, y_true = read_data("results.txt")
    acc = accuracy_score(y_true, y_pred)
    recc = recall_score(y_true, y_pred, average='macro')

    with open("current_results.txt", 'w') as f:
        f.write(f"accuracy: {acc} recall: {recc}")
    f.close()

    with open("metrics.txt", 'a') as f:
        f.write(f"{acc},{recc}\n")
    f.close()

# drawing a plot
def draw_plt():
    acc, recc = read_data("metrics.txt")
    no_of_entries = list(range(1, len(acc)+1))
    print(acc)
    print(recc)

    plt.plot(no_of_entries, acc, color='green', lw=2, label='Accuracy')
    plt.plot(no_of_entries, recc, color='blue', lw=2, label='Recall')
    plt.xlabel('Number of builds')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig("output.jpg")


new_metrics()
draw_plt()
