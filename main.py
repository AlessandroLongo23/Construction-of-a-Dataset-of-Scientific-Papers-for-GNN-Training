from WebScraper import WebScraper
from LatexData import LatexData
from PDFData import PDFData
from MatchingTool import MatchingTool
from Tester import Tester
from Plotter import Plotter

import os
from tqdm import tqdm
import json
import time
import copy

def download_files():
    download_folder = 'arxiv_downloads'
    web_scraper = WebScraper('unarXive_230324_open_subset')
    web_scraper.download_arxiv_papers("2404.15266", "2404.15270", download_folder)
    web_scraper.reorganize_unarXive_papers()
    web_scraper.download_unarXive_papers()

def single_compiling(start_index, end_index, search_id):
    papers = start_index if isinstance(start_index, list) else ([start_index] if end_index is None else [paper_id for paper_id in range(start_index, end_index + 1)])

    download_folder = 'arxiv_downloads'
    accuracies = []
    par_combination = json.load(open(f'search {search_id}/best_combination.json', 'r', encoding='utf-8'))
    for paper_id in papers:
        for folder_name in os.listdir(download_folder):
            folder_path = os.path.join(download_folder, folder_name)
            if int(folder_path[-2:]) == paper_id:
                # try:
                    start = time.time()
                    print(f'-' * 100 + f'\nProcessing {folder_path}:')
                    os.makedirs(os.path.join(folder_path, 'output'), exist_ok=True)

                    latex_data = LatexData(folder_path)
                    pdf_data = PDFData(folder_path)
                    matching_tool = MatchingTool(pdf_data, latex_data, folder_path, par_combination)
                    matching_tool.mark_pdf(pdf_data.elements['text_boxes'], 'processing')
                    pdf_results = matching_tool.perform_matching()
                    
                    pdf_data.export_to_json(pdf_results, 'pdf_data')
                    latex_data.export_to_json()
                    matching_tool.mark_pdf(pdf_results, 'output')
                    
                    try:
                        tester = Tester()
                        tester.load_ground_truth(folder_path)
                        accuracy = {
                            'id': '2404.152' + str(paper_id),
                            'accuracy': tester.evaluate_par_combination(pdf_results)
                        }
                    except:
                        print(f"Can't evaluate accuracy for {folder_path}")
                    
                    end = time.time()
                    print(f'2404.152{paper_id}: {accuracy['accuracy']:.2%} in {end - start:.2f} seconds')
                    accuracies.append(accuracy)
                    
                # except:
                #     print(f'Error processing {folder_path}')

    # paper_ids = np.arange(len(accuracies))
    # bar_width = 0.2
    # plt.figure(figsize=(10, 6))
    # plt.grid(True)
    # plt.bar(paper_ids, accuracies, bar_width, color='blue', zorder=2)
    # plt.xlabel('Paper ID')
    # plt.ylabel('Accuracy')
    # plt.title('Matching accuracy')
    # plt.xticks(paper_ids, ['2404.152' + str(p) for p in papers])
    # plt.tight_layout()
    # plt.show()

# def random_search(papers):
#     download_folder = 'arxiv_downloads'
#     tester = Tester()
#     c = 0
#     print(f'-' * 100)

#     par_combination = json.load(open('best_combination.json', 'r', encoding='utf-8'))
#     best_accuracies = [None] * len(papers)
#     for paper_id in papers:
#         for folder_name in os.listdir(download_folder):
#             folder_path = os.path.join(download_folder, folder_name)
#             if os.path.isdir(os.path.join(folder_path, 'output')) and int(folder_path[-2:]) == paper_id:
#                 latex_data = LatexData(folder_path)
#                 pdf_data = PDFData(folder_path) 
#                 matching_tool = MatchingTool(pdf_data, latex_data, folder_path, par_combination)
#                 pdf_results = matching_tool.perform_matching()
                
#                 tester.load_ground_truth(folder_path)
#                 best_accuracies[c] = tester.evaluate_par_combination(pdf_results)
#                 c += 1

#     par_combination['accuracy'] = sum(best_accuracies) / len(papers)

#     with open('best_combination.json', 'w') as file:
#         json.dump(par_combination, file)

#     iteration = 0
#     time_taken = []
#     average_time = []

#     while iteration < 50:
#         start_time = time.time()
#         iteration += 1
#         print(f'-' * 100)
#         next_par_combination = tester.next_par_combination(par_combination)
#         accuracies = [None] * len(papers)
#         c = 0
#         for paper_id in papers:
#             folder_path = os.path.join(download_folder, f'2404.152{paper_id}')
#             if os.path.isdir(os.path.join(folder_path, 'output')):
#                 latex_data = LatexData(folder_path)
#                 pdf_data = PDFData(folder_path) 
#                 matching_tool = MatchingTool(pdf_data, latex_data, folder_path, next_par_combination)
#                 pdf_results = matching_tool.perform_matching()
                
#                 tester.load_ground_truth(folder_path)
#                 accuracies[c] = tester.evaluate_par_combination(pdf_results)
#                 c += 1

#         next_par_combination['accuracy'] = sum(accuracies) / len(papers)

#         if par_combination is None or next_par_combination['accuracy'] > par_combination['accuracy']:
#             par_combination = next_par_combination
#             print(f'\nNew best combination found!\nbest combination accuracy: {par_combination["accuracy"] * 100:.2f}%')
#             for step in par_combination['operations']:
#                 print(step)

#             with open('best_combination.json', 'w') as file:
#                 json.dump(par_combination, file)
        
#         end_time = time.time()
#         time_taken.append(end_time - start_time)
#         average_time.append(sum(time_taken) / len(time_taken))

#     # plt.figure(figsize=(10, 6))
#     # plt.plot(time_taken, marker='o', linestyle='-', label='Time per iteration')
#     # plt.plot(average_time, marker='x', linestyle='--', label='Running average time')
#     # plt.xlabel('Iteration')
#     # plt.ylabel('Time taken (seconds)')
#     # plt.title('Time taken per iteration and running average time')
#     # plt.legend()
#     # plt.grid(True)
#     # plt.show()

def random_search_optimized(paper_ids, search_id):
    download_folder = 'arxiv_downloads'
    tester = Tester()
    plotter = Plotter(search_id)

    c = 0
    print(f'-' * 100)
    latex_datas = []
    pdf_datas = []

    try:
        par_combination = json.load(open(f'search {search_id}/best_combination.json', 'r', encoding='utf-8'))
    except FileNotFoundError:
        par_combination = tester.random_parameter_combination(13)

    best_accuracies = []
    for paper_id in paper_ids:
        for folder_name in os.listdir(download_folder):
            folder_path = os.path.join(download_folder, folder_name)
            if os.path.isdir(os.path.join(folder_path, 'output')) and int(folder_path[-2:]) == paper_id:
                latex_data = LatexData(folder_path)
                pdf_data = PDFData(folder_path)

                latex_datas.append({
                    'id': f'2404.152{paper_id}', 
                    'data': latex_data
                })
                pdf_datas.append({
                    'id': f'2404.152{paper_id}', 
                    'data': pdf_data
                })
                matching_tool = MatchingTool(pdf_data, latex_data, folder_path, par_combination)

                pdf_results = matching_tool.perform_matching()
                
                tester.load_ground_truth(folder_path)
                best_accuracies.append({
                    'id': f'2404.152{paper_id}', 
                    'accuracy': tester.evaluate_par_combination(pdf_results)
                })
                c += 1

    par_combination['accuracy'] = sum([acc['accuracy'] for acc in best_accuracies]) / len(paper_ids)  

    with open(f'search {search_id}/best_combination.json', 'w') as file:
        json.dump(par_combination, file)

    threshold = 0.01
    # iteration = 0
    # time_taken = []
    # average_time = []

    try:
        accuracies_history = json.load(open(f'search {search_id}/accuracies_history.json', 'r', encoding='utf-8'))
    except FileNotFoundError:
        accuracies_history = []

    try:
        best_combinations = json.load(open(f'search {search_id}/best_combination_history.json', 'r', encoding='utf-8'))
    except FileNotFoundError:
        best_combinations = [] 

    iteration = len(accuracies_history)
    while True:
        # start_time = time.time()
        print(f'-' * 100)
        next_par_combination = tester.next_par_combination(par_combination)
        accuracies = [{
            'id': f'2404.152{paper_id}',
            'accuracy': None
        } for paper_id in paper_ids]
        c = 0
        skip = False
        for paper_id in paper_ids:
            folder_path = os.path.join(download_folder, f'2404.152{paper_id}')
            if os.path.isdir(os.path.join(folder_path, 'output')):
                start = time.time()
                matching_tool = MatchingTool(pdf_datas[c]['data'], latex_datas[c]['data'], folder_path, next_par_combination)
                pdf_results = matching_tool.perform_matching()
                
                tester.load_ground_truth(folder_path)
                accuracies[c]['accuracy'] = tester.evaluate_par_combination(pdf_results)
                d = accuracies[c]['accuracy'] - best_accuracies[c]['accuracy']
                end = time.time()
                print(f'2404.152{paper_id}: {accuracies[c]['accuracy']:.2%} ({"+" if d > 0 else ""}{d * 100:.2f}%) in {end - start:.2f} seconds')

                c += 1

                if sum([acc['accuracy'] for acc in best_accuracies[:c]]) - sum([acc['accuracy'] for acc in accuracies[:c]]) > threshold: 
                    skip = True
                    break

        iteration += 1
        
        if not skip:
            next_par_combination['accuracy'] = sum(acc['accuracy'] for acc in accuracies) / len(paper_ids)

            if next_par_combination['accuracy'] >= par_combination['accuracy']:
                if next_par_combination['accuracy'] > par_combination['accuracy']:
                    print(f'\nNew best combination found with {par_combination["accuracy"] * 100:.2f}% average accuracy')

                best_accuracies = accuracies
                par_combination = next_par_combination
                with open(f'search {search_id}/best_combination.json', 'w') as file:
                        json.dump(par_combination, file)
                
        with open(f'search {search_id}/accuracies_history.json', 'w') as file:
            accuracies_history.append(best_accuracies)
            json.dump(accuracies_history, file)

        with open(f'search {search_id}/best_combination_history.json', 'w') as file:
            best_combinations.append(par_combination)
            json.dump(best_combinations, file)

        plotter.plot_accuracy_history(paper_ids, accuracies_history, 'accuracies_history')
        plotter.plot_parameters(best_combinations, 'best_combination_parameters')
        plotter.plot_sequence_parameters(best_combinations, 'sequence_parameters')

        # end_time = time.time()
        # time_taken.append(end_time - start_time)
        # average_time.append(sum(time_taken) / len(time_taken))

    # plt.figure(figsize=(10, 6))
    # plt.plot(time_taken, marker='o', linestyle='-', label='Time per iteration')
    # plt.plot(average_time, marker='x', linestyle='--', label='Running average time')
    # plt.xlabel('Iteration')
    # plt.ylabel('Time taken (seconds)')
    # plt.title('Time taken per iteration and running average time')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


# def brute_force_search(papers_ids):
#     download_folder = 'arxiv_downloads'
#     tester = Tester()
#     par_combinations = tester.generate_parameter_combinations(5, 0.25)

#     for par_combination in tqdm(par_combinations, desc=f'- searching for best combination'):
#         for folder_name in os.listdir(download_folder):
#             folder_path = os.path.join(download_folder, folder_name)
#             if os.path.isdir(folder_path) and int(folder_path[-2:]) in papers_ids:
#                 print(f'-' * 100 + f'\nProcessing {folder_path}:\n')
#                 os.makedirs(os.path.join(folder_path, 'output'), exist_ok=True)
#                 latex_data = LatexData(folder_path)
#                 pdf_data = PDFData(folder_path)
#                 matching_tool = MatchingTool(pdf_data, latex_data, folder_path)
#                 tester.load_ground_truth(folder_path)
                
#                 pdf_results = matching_tool.perform_matching(par_combination)
#                 sum_accuracy += tester.evaluate_par_combination(par_combination)

#         par_combination['accuracy'] = sum_accuracy / len(papers_ids)

#     best_combination = max(tester.parameter_combination, key=lambda x: x['accuracy'])
#     print(f'Best combination: {best_combination}')

def local_search(paper_ids, search_id):
    download_folder = 'arxiv_downloads'
    tester = Tester()
    plotter = Plotter(search_id)

    c = 0
    print(f'-' * 100)
    latex_datas = []
    pdf_datas = []

    par_combination = json.load(open(f'search {search_id}/best_combination.json', 'r', encoding='utf-8'))
    
    best_accuracies = []
    for paper_id in paper_ids:
        for folder_name in os.listdir(download_folder):
            folder_path = os.path.join(download_folder, folder_name)
            if os.path.isdir(os.path.join(folder_path, 'output')) and int(folder_path[-2:]) == paper_id:
                latex_data = LatexData(folder_path)
                pdf_data = PDFData(folder_path)

                latex_datas.append({
                    'id': f'2404.152{paper_id}', 
                    'data': latex_data
                })
                pdf_datas.append({
                    'id': f'2404.152{paper_id}', 
                    'data': pdf_data
                })
                matching_tool = MatchingTool(pdf_data, latex_data, folder_path, par_combination)
                pdf_results = matching_tool.perform_matching()
                
                tester.load_ground_truth(folder_path)
                best_accuracies.append({
                    'id': f'2404.152{paper_id}', 
                    'accuracy': tester.evaluate_par_combination(pdf_results)
                })
                c += 1

    par_combination['accuracy'] = sum(acc['accuracy'] for acc in best_accuracies) / len(paper_ids)

    with open(f'search {search_id}/best_combination.json', 'w') as file:
        json.dump(par_combination, file)

    threshold = 0.01
    while True:
        # parameters = ['num_nearest_lines', 'num_nearest_blocks', 'context_words', 'siblings_offset', 'cousins_offset', 'equation_dist_threshold', 'lcs_max_distance']
        # for parameter in parameters:
        #     local_combinations = tester.generate_local_combinations(par_combination, parameter)
        #     for local_combination in tqdm(local_combinations, desc=f'- searching for best {parameter} local combination'):
        #         accuracies = [{
        #             'id': f'2404.152{paper_id}',
        #             'accuracy': None
        #         } for paper_id in paper_ids]
        #         c = 0
        #         skip = False
        #         for paper_id in paper_ids:
        #             folder_path = os.path.join(download_folder, f'2404.152{paper_id}')
        #             if os.path.isdir(os.path.join(folder_path, 'output')):
        #                 start = time.time()
        #                 matching_tool = MatchingTool(pdf_datas[c]['data'], latex_datas[c]['data'], folder_path, local_combination)
        #                 pdf_results = matching_tool.perform_matching()
                        
        #                 tester.load_ground_truth(folder_path)
        #                 accuracies[c]['accuracy'] = tester.evaluate_par_combination(pdf_results)
        #                 d = accuracies[c]['accuracy'] - best_accuracies[c]['accuracy']
        #                 end = time.time()
        #                 print(f'2404.152{paper_id}: {accuracies[c]['accuracy']:.2%} ({"+" if d > 0 else ""}{d * 100:.2f}%) in {end - start:.2f} seconds')
        #                 c += 1

        #                 if sum([acc['accuracy'] for acc in best_accuracies[:c]]) - sum([acc['accuracy'] for acc in accuracies[:c]]) > threshold: 
        #                     skip = True
        #                     break

        #         if not skip:
        #             local_combination['accuracy'] = sum(acc['accuracy'] for acc in accuracies) / len(paper_ids)

        #             if local_combination['accuracy'] > par_combination['accuracy']:
        #                 par_combination = local_combination
        #                 best_accuracies = accuracies
        #                 print(f'\nNew best combination found!\nbest combination accuracy: {par_combination["accuracy"] * 100:.2f}%')

        #                 with open(f'search {search_id}/best_combination.json', 'w') as file:
        #                     json.dump(par_combination, file)

        for i in range(len(par_combination['operations'])):
            if par_combination['operations'][i][0] != 'equation' and par_combination['operations'][i][0] != 'leftover':
                for j in [1, 2]:
                    local_combinations = tester.generate_local_combinations(par_combination, 'sequence', i, j)
                    for local_combination in tqdm(local_combinations, desc=f'- searching for best local combination'):
                        accuracies = [{
                            'id': f'2404.152{paper_id}',
                            'accuracy': None
                        } for paper_id in paper_ids]
                        c = 0
                        skip = False
                        for paper_id in paper_ids:
                            folder_path = os.path.join(download_folder, f'2404.152{paper_id}')
                            if os.path.isdir(os.path.join(folder_path, 'output')):
                                start = time.time()
                                matching_tool = MatchingTool(pdf_datas[c]['data'], latex_datas[c]['data'], folder_path, local_combination)
                                pdf_results = matching_tool.perform_matching()
                                
                                tester.load_ground_truth(folder_path)
                                accuracies[c]['accuracy'] = tester.evaluate_par_combination(pdf_results)
                                d = accuracies[c]['accuracy'] - best_accuracies[c]['accuracy']
                                end = time.time()
                                print(f'2404.152{paper_id}: {accuracies[c]['accuracy']:.2%} ({"+" if d > 0 else ""}{d * 100:.2f}%) in {end - start:.2f} seconds')
                                c += 1

                                if sum([acc['accuracy'] for acc in best_accuracies[:c]]) - sum([acc['accuracy'] for acc in accuracies[:c]]) > threshold: 
                                    skip = True
                                    break

                        if not skip:
                            local_combination['accuracy'] = sum(acc['accuracy'] for acc in accuracies) / len(paper_ids)

                            if local_combination['accuracy'] > par_combination['accuracy']:
                                par_combination = local_combination
                                best_accuracies = accuracies
                                print(f'\nNew best combination found!\nbest combination accuracy: {par_combination["accuracy"] * 100:.2f}%')

                                with open(f'search {search_id}/best_combination.json', 'w') as file:
                                    json.dump(par_combination, file)

        try:
            accuracies_history = json.load(open(f'search {search_id}/accuracies_history.json', 'r', encoding='utf-8'))
        except FileNotFoundError:
            accuracies_history = []

        try:
            best_combinations = json.load(open(f'search {search_id}/best_combination_history.json', 'r', encoding='utf-8'))
        except FileNotFoundError:
            best_combinations = [] 
        
        with open(f'search {search_id}/accuracies_history.json', 'w') as file:
            accuracies_history.append(best_accuracies)
            json.dump(accuracies_history, file)

        with open(f'search {search_id}/best_combination_history.json', 'w') as file:
            best_combinations.append(par_combination)
            json.dump(best_combinations, file)

        plotter.plot_accuracy_history(paper_ids, accuracies_history, 'accuracies_history')
        plotter.plot_parameters(best_combinations, 'best_combination_parameters')
        plotter.plot_sequence_parameters(best_combinations, 'sequence_parameters')

if __name__ == "__main__":
    # single_compiling(20, 20, 2)
    random_search_optimized([29, 34, 25, 19, 67], 2)
    # local_search([29, 34, 25, 19, 67], 2)
    # random_search([19, 25, 29, 34])

