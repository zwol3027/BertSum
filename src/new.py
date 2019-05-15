main_directory = os.getcwd()

os.system('touch logs/cnndm.log')
os.system('touch logs/preprocess.log')

os.system('rm -r merged_stories_tokenized; mkdir merged_stories_tokenized')
os.system('rm -r json_data; mkdir json_data')
os.system('rm -r bert_data; mkdir bert_data')
os.system('rm -r results; mkdir results')


os.chdir(main_directory +'/src')

os.system('python preprocess.py -mode tokenize -raw_path ../raw_text_files -save_path ../merged_stories_tokenized -log_file ../logs/cnndm.log')
os.system('python preprocess.py -mode format_to_lines_NEW -raw_path ../merged_stories_tokenized -n_cpus 1 -save_path ../json_data/cnndm -map_path ../urls -lower -dataset test -log_file ../logs/cnndm.log')
os.system ('python preprocess.py -mode format_to_bert -dataset test -raw_path ../json_data -save_path ../bert_data -oracle_mode greedy -n_cpus 1 -log_file ../logs/preprocess.log')


os.system('python train.py -mode test -bert_data_path ../bert_data/cnndm -test_from ../pretrained_bertsum/cnndm_bertsum_classifier_best.pt -visible_gpus 0  -gpu_ranks 0 -batch_size 30000 -result_path ../results/tester -block_trigram true -report_rouge False')

os.chdir(main_directory)

with open('results/tester_step0.candidate', 'r') as file:
    summary = file.read().replace('\n', '')
    
summary = summary.replace('<q>', '\n')
summary = summary.replace(' ,', ',')
summary = summary.replace(' .', '.')

print(summary)
