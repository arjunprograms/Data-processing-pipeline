# Arjun Subedi
# 8/20/24
# Purpose: Script to run the pipeline
# Dependencies: code/base_component.py, code/helper.py

# Imports (Import all classes/scripts here)

# Imports (already provided in the script)
from bio_pipeline_package.base_component import Block
from bio_pipeline_package import component_library
from multiprocessing import Process
import os
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler("pipeline.log", maxBytes=1000000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[handler, logging.StreamHandler()])

# Pipeline Blocks Configuration
blocks = {
    1: {  # CSV_Pandas Block
        'class': True,
        'script': False,
        'triggered_start': True,
        'class_name': 'CSV_Pandas',
        'input_ids': [],
        'input_files': ["inputs/Byeon2022_data.csv"],
        'output_ids': [2],
        'output_files': [],
        'obj': None,
        'args': (),
    },
    2: {  # PCAnalysis Block
        'class': True,
        'script': False,
        'triggered_start': False,
        'class_name': 'KMeansCluster',
        'input_ids': [1],  # Receives input from FeatureSelection Block
        'input_files': [],
        'output_ids': [],
        'output_files': [],
        'obj': None,
        'args': (4, 1),  # Number of principal components and target column
    },
}



def main():
    # Delete files in comms_files directory before starting (if needed)
    logging.info("Attempting to delete files in 'comms_files' directory...")
    comms_dir = 'comms_files'
    if os.path.exists(comms_dir):
        for f in os.listdir(comms_dir):
            os.remove(os.path.join(comms_dir, f))
    else:
        os.mkdir(comms_dir)
    
    # Initialize each block
    m_list = []
    for id, block in blocks.items():
        in_regex = []  # Initialize in_regex list
        out_stubs = []  # Initialize out_stubs list
        
        for inp in block['input_ids']:
            in_regex.append(r'.*_' + str(id) + r'\..*')
        for out in block['output_ids']:
            out_stubs.append(os.path.join('comms_files', str(id) + '_' + str(out)))

        block['in_regex'] = in_regex
        block['out_stubs'] = out_stubs

        if 'class' in block and block['class']:
            Found_Class = getattr(component_library, block['class_name'])
            block['obj'] = Found_Class(block)
            m_list.append(block['obj'])
            logging.info(f"Initialized block {id}: {block['class_name']}")
    
    # Start multiprocessing for each block
    active_processes = []
    logging.info('Starting multiprocessing for the following blocks: %s', m_list)
    for p in m_list:
        new_p = Process(target=p.listen, args=())
        new_p.start()
        active_processes.append({'type': 'multiprocess', 'process': new_p})
        logging.info(f"Started process for block: {p}")
    
    # Wait for user input to stop the processes
    done = False
    while not done:
        user_in = input("Press 's' and then enter when ready to stop: ")
        if user_in == 's':
            done = True
        else:
            logging.info("Did not enter 's'")
    
    # Terminate all active processes
    for active_process in active_processes:
        if 'type' in active_process and active_process['type'] == 'multiprocess':
            logging.info("Stopping process for block.")
            active_process['process'].terminate()

if __name__ == '__main__':
    # Required for Windows to prevent recursive spawning
    import multiprocessing
    multiprocessing.freeze_support()
    logging.info("Starting main function.")
    main()