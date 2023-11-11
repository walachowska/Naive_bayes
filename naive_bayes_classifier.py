import pandas


class NaiveBayesClassifier:
    def __init__(self, train_data, test_data, decisions):
        self.train_set = train_data
        self.test_set = test_data
        self.decisions = decisions

    def calculate_probabilities(self, row) -> dict:
        attributes = row[:6].values
        print(f'Certain attributes: {attributes}')
        # dict for 4 results
        probabilities = {}
        for decision in self.decisions:
            # decisions probabilities P(Y)
            num_of_all_decisions = len(self.train_set['class'])
            decision_probability = (self.train_set['class'] == decision).sum() / num_of_all_decisions
            # start counting from probability from decision probability
            final_probability = decision_probability
            # count probabilities for attributes

            for attribute, column in zip(attributes, self.train_set.columns[:-1]):
                attribute_occ = ((self.train_set[column] == attribute) & (self.train_set['class'] == decision)).sum()
                decision_att_occ = (self.train_set['class'] == decision).sum()
                # whether smoothing is necessary
                if attribute_occ == 0:
                    attribute_occ += 1
                    num_of_unique_att = self.train_set[column].nunique()
                    decision_att_occ += num_of_unique_att
                attribute_probability = attribute_occ / decision_att_occ
                final_probability *= attribute_probability

            probabilities[decision] = final_probability

        return probabilities

    def classify(self) -> int:
        correct_classifications = 0
        for _, row in self.test_set.iterrows():
            # dict for results of 4 decision attributes
            probabilities = self.calculate_probabilities(row)
            # print(f'Results: {probabilities}')
            best_result = max(probabilities, key=probabilities.get)
            print(f'These attributes were classified as {best_result}')
            expected_class = row['class']
            print(f'Real Value: {expected_class}')
            # note correct classification
            if best_result == expected_class:
                correct_classifications += 1
        return correct_classifications

    def show_accuracy(self, correct_anwsers) -> None:
        print('\nNumber of correct classifications: ', correct_anwsers)
        num_of_all_classifications = len(self.test_set)
        accuracy_percentage = (correct_anwsers / num_of_all_classifications) * 100
        print(f'Accuracy: {accuracy_percentage} %')


# read data
data = pandas.read_csv('car_evaluation.data', header=None)
data.columns = ['price', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
# data mixing
data = data.sample(frac=1).reset_index(drop=True)
# setting test set size
test_size = int(len(data) * 0.30)
# partition to test 30%  and train sets 70%
test_set = data[:test_size].reset_index(drop=True)
train_set = data[test_size:].reset_index(drop=True)
# decision attributes
decisions = ['unacc', 'acc', 'vgood', 'good']
# classification
classifier = NaiveBayesClassifier(train_set, test_set, decisions)
correct_classifications = classifier.classify()
# results
print(f"\nResults for a {test_size}x test size: ")
classifier.show_accuracy(correct_classifications)


