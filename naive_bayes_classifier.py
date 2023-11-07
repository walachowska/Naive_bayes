import pandas


class NaiveBayesClassifier:
    def __init__(self, data, decisions):
        self.data = data
        self.decisions = decisions

    def calculate_probabilities(self, row) -> dict:
        attributes = row[:6].values
        print(f'Certain attributes: {attributes}')
        # dict for 4 results
        probabilities = {}
        for decision in self.decisions:
            # decisions probabilities P(Y)
            num_of_all_decisions = len(self.data['class'])
            decision_probability = (self.data['class'] == decision).sum() / num_of_all_decisions
            # start counting from probability from decision probability
            final_probability = decision_probability
            # count probabilities for attributes

            for attribute, column in zip(attributes, self.data.columns[:-1]):
                attribute_occ = ((self.data[column] == attribute) & (self.data['class'] == decision)).sum()
                decision_att_occ = (self.data['class'] == decision).sum()
                # whether smoothing is necessary
                if attribute_occ == 0:
                    attribute_occ += 1
                    num_of_unique_att = self.data[column].nunique()
                    decision_att_occ += num_of_unique_att
                attribute_probability = attribute_occ / decision_att_occ
                final_probability *= attribute_probability

            probabilities[decision] = final_probability

        return probabilities

    def classify(self) -> int:
        correct_classifications = 0
        for _, row in self.data.iterrows():
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

    def show_accuracy(self, correct_anwsers)-> None:
        print('\nNumber of correct classifications: ', correct_anwsers)
        num_of_rows = len(self.data)
        accuracy_percentage = (correct_anwsers / num_of_rows) * 100
        print(f'Accuracy: {accuracy_percentage} %')


# read data
data = pandas.read_csv('car_evaluation.data', header=None)
data.columns = ['price', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
# decision attributes
decisions = ['unacc', 'acc', 'vgood', 'good']
# classification
bayes = NaiveBayesClassifier(data, decisions)
correct_classifications = bayes.classify()
# results
bayes.show_accuracy(correct_classifications)

