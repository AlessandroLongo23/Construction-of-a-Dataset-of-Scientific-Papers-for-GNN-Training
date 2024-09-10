import json
import numpy as np
import os
import random
import copy

class Tester():
    def load_ground_truth(self, folder_path):
        with open(os.path.join(folder_path, 'output', 'pdf_ground_truth.json'), 'r', encoding='utf-8') as file:
            data = json.load(file)

        self.ground_truth = data

    def evaluate_par_combination(self, pdf_lines):
        correct = 0
        for pdf_line in pdf_lines:
            ground_truth_line = None
            for item in self.ground_truth:
                if item['page'] == pdf_line.page_index and item['index'] == pdf_line.index:
                    ground_truth_line = item
                    break

            if ground_truth_line is None:
                continue

            if len(ground_truth_line['matches']) == 0 or ground_truth_line['matches'][0][0] == -1 or (pdf_line.assigned and pdf_line.matches[0][0] == ground_truth_line['matches'][0][0]):
                correct += 1

        return correct / len(pdf_lines)

    def next_par_combination(self, par_combination=None):
        if par_combination is None:
            return self.random_parameter_combination()

        next_par_combination = copy.deepcopy(par_combination)
        for i, step in enumerate(next_par_combination['operations']):
            if step[0] == 'equation' and random.random() < 0.1:
                move = int(np.sign(random.random() - 0.5))
                if (move == 1 and i < len(next_par_combination['operations']) - 2) or (move == -1 and i > 0):
                    temp = next_par_combination['operations'][i + move]
                    next_par_combination['operations'][i] = next_par_combination['operations'][i + move]
                    next_par_combination['operations'][i + move] = temp

        for i, step in enumerate(next_par_combination['operations']):
            if next_par_combination['operations'][i][0] == 'best_guess' or next_par_combination['operations'][i][0] == 'accurate':
                if random.random() < 0.05:
                    if random.random() < 0.5:
                        next_par_combination['operations'].remove(next_par_combination['operations'][i])
                    else:
                        next_par_combination['operations'].insert(i, [
                            'accurate' if random.random() < 0.5 else 'best_guess',
                            random.random(),
                            random.random(),
                            next_par_combination['operations'][i][3]
                        ])

        variance = 0.002
        for i, step in enumerate(next_par_combination['operations']):
            if next_par_combination['operations'][i][0] == 'best_guess' or next_par_combination['operations'][i][0] == 'accurate':
                next_par_combination['operations'][i][1] = round(min(max(next_par_combination['operations'][i][1] + random.gauss(0, variance), 0), 1), 4)
                next_par_combination['operations'][i][2] = round(min(max(next_par_combination['operations'][i][2] + random.gauss(0, variance), 0), 1), 4)
                if random.random() < 0.025:
                    next_par_combination['operations'][i][0] = 'best_guess' if next_par_combination['operations'][i][0] == 'accurate' else 'accurate'

        if random.random() < 0.05: next_par_combination['num_nearest_lines'] = max(next_par_combination['num_nearest_lines'] + random.randint(-1, 1), 0)
        if random.random() < 0.05: next_par_combination['num_nearest_blocks'] = max(next_par_combination['num_nearest_blocks'] + random.randint(-1, 1), 0)
        if random.random() < 0.05: next_par_combination['context_words'] = max(next_par_combination['context_words'] + random.randint(-1, 1), 0)
        if random.random() < 0.05: next_par_combination['siblings_offset'] = max(next_par_combination['siblings_offset'] + random.randint(-1, 1), 0)
        if random.random() < 0.05: next_par_combination['cousins_offset'] = max(next_par_combination['cousins_offset'] + random.randint(-1, 1), 0)
        if random.random() < 0.05: next_par_combination['equation_dist_threshold'] = max(next_par_combination['equation_dist_threshold'] + random.randint(-1, 1), 0)
        if random.random() < 0.05: next_par_combination['lcs_max_distance'] = max(next_par_combination['lcs_max_distance'] + random.randint(-1, 1), 0)

        return next_par_combination

    def random_parameter_combination(self, num_operations):
        operations = []
        equation_aggregation_index = random.randint(0, num_operations - 1)
        for i in range(num_operations - 1):
            if i == equation_aggregation_index:
                operations.append(['equation'])
            else:
                op_type = 'accurate' if random.random() < 0.5 else 'best_guess'
                threshold = random.random()
                difference = random.random()
                operations.append([op_type, threshold, difference, i < equation_aggregation_index])

        operations.append(['leftover'])

        return {
            "operations": operations,
            "num_nearest_lines": random.randint(3, 7),
            "num_nearest_blocks": random.randint(2, 6),
            "context_words": random.randint(1, 5),
            "siblings_offset": random.randint(1, 4),
            "cousins_offset": random.randint(15, 21),
            "equation_dist_threshold": random.randint(12, 20),
            "lcs_max_distance": random.randint(28, 34),
            "accuracy": None
        }

    def parameter_combination(self, i, j, k):
        operations = []
        equation_aggregated = True
        for l in range(self.num_operations - 1):
            if l == j:
                operations.append(['equation'])
                equation_aggregated = False
            else:
                op_type = k % 2
                k = (k - op_type) / 2
                threshold = i % self.steps
                i = (i - threshold) / self.steps
                difference = i % self.steps
                i = (i - difference) / self.steps
                operations.append(['accurate' if op_type == 0 else 'best_guess', threshold / (self.steps - 1), difference / (self.steps - 1), equation_aggregated])

        operations.append(['leftover'])

        return {
            "operations": operations,
            "num_nearest_lines": 5,
            "num_nearest_blocks": 4,
            "context_words": 3,
            "siblings_offset": 2,
            "cousins_offset": 18,
            "equation_dist_threshold": 16,
            "lcs_max_distance": 31,
            "accuracy": None
        }
    
    def generate_local_combinations(self, par_combination, parameter, i=None, v=None):
        local_combinations = []

        if parameter == 'sequence':
            for j in range(-6, 7, 4):
                local_combination = copy.deepcopy(par_combination)
                local_combination['operations'][i][v] = round(min(max(local_combination['operations'][i][v] + j / 100, 0), 1), 4)
                local_combinations.append(local_combination)
        else:
            for off in [-2, -1, 1,2]:
                local_combination = copy.deepcopy(par_combination)
                local_combination[parameter] = max(local_combination[parameter] + off, 0)
                local_combinations.append(local_combination)
        
        return local_combinations

    # def generate_parameter_combinations(self, num_operations, step):
    #     self.num_operations = num_operations
    #     self.steps = int(1 / step) + 1
    
    #     par_combinations = []
    #     for i in range(self.steps ** (2 * (self.num_operations - 2))):
    #         for j in range(self.num_operations - 1):
    #             for k in range(2 ** (self.num_operations - 2)):
    #                 par_combinations.append(self.parameter_combination(i, j, k))

    #     return par_combinations

    