import pandas as pd
# read data
data = pd.read_csv('car_evaluation.data', header=None)
data.columns = ['price', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
# decision attributes
list_of_decisions = ['unacc', 'acc', 'vgood', 'good']
correct_classifications = 0
for _, row in data.iterrows():
    # wanted attributes = ['low', 'high', '2', '4', 'big', 'high']
    attributes = row[:6].values
    print(f'Certain attributes: {attributes}')
    # dict for 4 results
    probabilities = {}
    # count probabilities for all decision attributes
    for decision in list_of_decisions:
        # decisions probabilities P(Y)
        num_of_all_decisions = len(data['class'])
        decision_probability = (data['class'] == decision).sum() / num_of_all_decisions
        # start counting from probability from decision probability
        final_probability = decision_probability
        # count probabilities for attributes
        for attribute, column in zip(attributes, data.columns[:-1]):
            attribute_occ = ((data[column] == attribute) & (data['class'] == decision)).sum()
            decision_att_occ = (data['class'] == decision).sum()
            # whether smoothing is necessary
            if attribute_occ == 0:
                attribute_occ += 1
                num_of_unique_att = data[column].nunique()
                decision_att_occ += num_of_unique_att
            attribute_probability = attribute_occ / decision_att_occ
            final_probability *= attribute_probability
        probabilities[decision] = final_probability
    # print(f'Results: {probabilities}')
    best_result = max(probabilities, key=probabilities.get)
    print(f'These attributes were classified as {best_result}')
    expected_class = row['class']
    print(f'Real Value: {expected_class}')
    # note correct classification
    if best_result == expected_class:
        correct_classifications += 1
num_of_rows = len(data)
print('Number of correct classifications: ', correct_classifications)
accuracy_percentage = (correct_classifications / num_of_rows) * 100
print(f'Accuracy: {accuracy_percentage} %')