import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, search_id):
        self.search_id = search_id

    def plot_accuracy_history(self, paper_ids, accuracies_history, savename=None):
        plt.figure(figsize=(10, 6))
        for paper_id in ['2404.152' + str(paper) for paper in paper_ids]:
            graph = []
            for acc_list in accuracies_history:
                for item in acc_list:
                    if item['id'] == paper_id:
                        graph.append(item['accuracy'])
            plt.plot(graph, marker='', linestyle='-', label=paper_id)
        plt.plot([sum([value['accuracy'] for value in acc if int(value['id'][-2:]) in paper_ids]) / len(acc) for acc in accuracies_history], marker='', color='black', linestyle='--', label='average')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Best accuracy per iteration')
        plt.legend()
        plt.grid(True)
        if savename is not None:
            plt.savefig(f'search {self.search_id}/{savename}.png')
        plt.close()

    def plot_parameters(self, best_combinations, savename=None):
        plt.figure(figsize=(10, 6))
        parameters = ['num_nearest_lines', 'num_nearest_blocks', 'context_words', 'siblings_offset', 'cousins_offset', 'equation_dist_threshold', 'lcs_max_distance']
        for parameter in parameters:
            plt.plot([cmb[parameter] for cmb in best_combinations], marker='', linestyle='-', label=parameter)
        plt.plot([len(cmb['operations']) for cmb in best_combinations], marker='', linestyle='-', label='num_operations')
        plt.xlabel('Parameter')
        plt.ylabel('Value')
        plt.title('Best combination parameters')
        plt.legend()
        plt.grid(True)
        if savename is not None:
            plt.savefig(f'search {self.search_id}/{savename}.png')
        plt.close()

    def plot_sequence_parameters(self, best_combinations, savename=None):
        plt.figure(figsize=(10, 6))

        for k in [1, 2]:
            parameters_trends = []
            for i in range(len(best_combinations[0]['operations'])):
                if best_combinations[0]['operations'][i][0] != 'equation' and best_combinations[0]['operations'][i][0] != 'leftover':
                    parameters_trends.append([best_combinations[0]['operations'][i][k]])

            for parameter_trend in parameters_trends:
                for i in range(len(best_combinations)):
                    diff = float('inf')
                    closest = None
                    for j in range(len(best_combinations[i]['operations'])):
                        if best_combinations[i]['operations'][j][0] != 'equation' and best_combinations[i]['operations'][j][0] != 'leftover' and abs(best_combinations[i]['operations'][j][k] - parameter_trend[-1]) < diff:
                            diff = abs(best_combinations[i]['operations'][j][k] - parameter_trend[-1])
                            closest = j
                    parameter_trend.append(best_combinations[i]['operations'][closest][k])

                plt.plot(parameter_trend, marker='', linestyle='-')

        # for i in range(len(best_combinations[0]['operations'])):
        #     for j in [1, 2]:
        #         plt.plot([comb['operations'][i][j] for comb in best_combinations if (comb['operations'][i][0] != 'equation' and comb['operations'][i][0] != 'leftover')], marker='', linestyle='-') 

        plt.xlabel('Parameter')
        plt.ylabel('Value')
        plt.title('Best combination parameters')
        plt.grid(True)
        if savename is not None:
            plt.savefig(f'search {self.search_id}/{savename}.png')
        plt.close()
