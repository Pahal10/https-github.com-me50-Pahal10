import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    month = ['Jan', 'Feb', 'Mar' , 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_num = enumerate(month)
    months = {k: v for v, k in month_num}

    with open(filename, "r") as f:
        reader = csv.DictReader(f)
      #  next(reader)

        for row in reader:
            a = 0
            b = 0
            c = 0
            d = 0

            if row["VisitorType"] == 'Returning_Visitor':
                a += 1
            if row["Weekend"] == 'TRUE':
                b += 1
            if row["Revenue"] == 'TRUE':
                c += 1
            for i in range(0, len(month)):
                if month[i] == row["Month"]:
                    d = i

            ev = []
            ev.append(int(row["Administrative"]))
            ev.append(float(row["Administrative_Duration"]))
            ev.append(int(row["Informational"]))
            ev.append(float(row["Informational_Duration"]))
            ev.append(int(row["ProductRelated"]))
            ev.append(float(row["ProductRelated_Duration"]))
            ev.append(float(row["BounceRates"]))
            ev.append(float(row["ExitRates"]))
            ev.append(float(row["PageValues"]))
            ev.append(float(row["SpecialDay"]))
            ev.append(int(d))
            ev.append(int(row["OperatingSystems"]))
            ev.append(int(row["Browser"]))
            ev.append(int(row["Region"]))
            ev.append(int(row["TrafficType"]))
            ev.append(int(a))
            ev.append(int(b))
            evidence.append(ev)

            labels.append(c)

    return evidence, labels



def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors = 3)
    model.fit(evidence, labels)
    return model



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    total = len(labels)
    pos = 0
    neg = 0
    true_pos = 0
    true__neg = 0

    for i in range(0, total):
        if labels[i] == predictions[i]:
            if labels[i] == 0:
                neg +=1
                true__neg += 1
            if labels[i] == 1:
                pos += 1
                true_pos += 1
        else:
            if labels[i] == 0:
                neg += 1
            if labels[i] == 1:
                pos += 1

    sensitivity = true_pos / pos
    specificity = true__neg / neg
    

    return sensitivity, specificity



if __name__ == "__main__":
    main()
