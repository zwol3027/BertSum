def main(RAW_TEXT):
  main_directory = os.getcwd()

  os.system('touch logs/cnndm.log')
  os.system('touch logs/preprocess.log')

  os.system('rm -r merged_stories_tokenized; mkdir merged_stories_tokenized')
  os.system('rm -r json_data; mkdir json_data')
  os.system('rm -r bert_data; mkdir bert_data')
  os.system('rm -r results; mkdir results')

  os.system('rm -r raw_text_files; mkdir raw_text_files')
  os.system('touch raw_text_files/input.story')

#   RAW_TEXT = input("Enter the text to summarize: ")

  f = open('raw_text_files/input.story', 'wt', encoding='utf-8')
  f.write(RAW_TEXT)
  f.close()

  os.chdir(main_directory +'/src')

  os.system('python preprocess.py -mode tokenize -raw_path ../raw_text_files -save_path ../merged_stories_tokenized -log_file ../logs/cnndm.log')
  os.system('python preprocess.py -mode format_to_lines_NEW -raw_path ../merged_stories_tokenized -n_cpus 1 -save_path ../json_data/cnndm -map_path ../urls -lower -dataset test -log_file ../logs/cnndm.log')
  os.system ('python preprocess.py -mode format_to_bert -dataset test -raw_path ../json_data -save_path ../bert_data -oracle_mode greedy -n_cpus 1 -log_file ../logs/preprocess.log')

  # Visible GPUS = -1 for CPU
  os.system('python train.py -mode test -bert_data_path ../bert_data/cnndm -test_from ../pretrained_bertsum/cnndm_bertsum_classifier_best.pt -visible_gpus -1  -gpu_ranks 0 -batch_size 30000 -result_path ../results/tester -block_trigram true -report_rouge False')

  os.chdir(main_directory)

  with open('results/tester_step0.candidate', 'r') as file:
      summary = file.read().replace('\n', '')

  summary = summary.replace('<q>', ' ')
  summary = summary.replace(' ,', ',')
  summary = summary.replace(' .', '.')
  return summary
  
scrum_text = 'Scrum is not an acronym, the name is taken from the sport of Rugby. It brings the analogy where Team works together to successfully develop software. Scrum is a iterative and incremental framework for application or product development. The development of the project is achieved through iterative cycles known as Sprints. At the start of each Sprint, a cross-functional team selects items from Product Backlog and commits to complete the items by the end of that particular Sprint. Everyday the team gathers for a short meeting to discuss the progress and road blocks if any. At the end of the Sprint the team reviews the work product with stakeholders and demonstrates what has been built. The feedback is then incorporated in the subsequent Sprint. At the end of each Sprint Scrum emphasizes that the integrated working software is fully tested and potentially made shippable. The Sprints are strictly time-boxed and occur sequentially. The end date of a Sprint does not get extended regardless of the completion of the work initially planned.'
print(main(scrum_text))
