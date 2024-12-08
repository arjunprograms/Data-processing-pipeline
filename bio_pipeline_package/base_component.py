#Arjun
#8/20/24
#Base block to inherit from when creating custom Python ones.
#Implement the same interface in R for compatibility.

#Imports
import bio_pipeline_package.helper
from collections import deque

import os
import logging
import time
import re

#File Handling Imports (Comment out any you don't need)
import pandas as pd
import json


class Block():
    #Files to listen to, output to, and whether it runs on startup.
    #Also the formats used.
    
    def __init__(self, block_config):
        self.bc = block_config
        self.input_queue = deque([])
        self.output_queue = deque([])
        self.arg_queue = deque([])
        
        
        #Customizable init
        self.extra_init()
        
    def extra_init(self):
        pass
    
    def read_file(self, input_path):
        #Reads from input file and converts it to desired format.
        print(self.bc['class_name'], "reading file: ", input_path)
        #f = open(inp, 'r').read()
        
        print("Input, working, output:  ", self.input_type, self.working_type, self.output_type)
        
        if self.input_type == 'csv':
            if self.working_type == 'pandas_df':
                #Check that it is the correct file type (TODO)
                df = pd.read_csv(input_path)
                self.input_queue.appendleft(df)
            
        elif self.input_type == 'pickle':
            if self.working_type == 'pandas_df':
               df = pd.read_pickle(input_path)
               self.input_queue.appendleft(df)
        
        elif self.input_type == 'json':
            if self.working_type == 'dict':
                with open(input_path, 'r') as f:
                    self.input_queue.appendleft(json.load(f))
            
        else:
            print(self.bc['class_name'], "reports incompatible or unknown input/working pair")
            
    def collect_args(self):
        '''
        collect arguments
        '''
        if self.bc['args'] == ():
            print("No arguments, skipping argument collection")
            self.arg_queue.appendleft(())
        else:
            self.arg_queue.appendleft(self.bc['args'])


    def write_file(self, entry, output_path):
        #Writes to output file
        
        #Check for stub (no file type)
        print("Raw output Path: ", output_path)
        root, ext = os.path.splitext(output_path)
        if not ext:
            print("Stub detected")
            if self.output_type == 'pickle': ext = 'pkl'
            if self.output_type == 'json': ext = 'json'
            if self.output_type == 'csv': ext = 'csv'
            if self.output_type == 'png': ext = 'png'
            output_path = root + '.' + ext
        print("Full output path: ", output_path)
        
        
        
        if self.working_type == 'pandas_df':
            if self.output_type == 'pickle':
                #Pickle it
                entry.to_pickle(output_path)
                
        
            if self.output_type == 'json':
                #Should have pandas to json conversion here (TODO)
                #Entry should be json or dict in this case
                print("Trying to write json")
                print("Entry is: ", entry)
                
                with open(output_path, 'w') as outfile:   
                    json.dump(entry, outfile)                
                '''
                if isinstance(entry, dict):
                    with open(output_path, 'w') as outfile:   
                        json.dump(entry, outfile)
                else:
                    print("Incompatible with json")
                pass   
                '''
                print("")

            if self.output_type == 'png':
                print("Trying to save png")
                print("Entry is: ", entry)

        elif self.working_type == 'dict':
            if self.output_type == 'json':
                #Entry should be json or dict in this case
                print("Trying to write json")
                print("Entry is: ", entry)
                
                with open(output_path, 'w') as outfile:   
                    json.dump(entry, outfile)
                    
                print("")

        
        else:
            print("Incompatible or unknown working/output pair")
        
    def listen(self):
        #--------Check or listen to files------------------
        if self.bc['triggered_start']:
            #print(self.bc['class_name'], "checking for input files")
            
            #Read in input files straight away.
            for inp in self.bc['input_files']:
                print(self.bc['class_name'], "checking input: ", inp)
                
                #Check if file
                if os.path.isfile(inp):
                    #---------Process Method-------------
                    self.read_file(inp)
                    self.collect_args()
                    #Process the queue
                    while len(self.input_queue) > 0:
                        #Pop off an entry
                        #Also pop off arguments
                        entry = self.input_queue.pop()
                        print('this is arg queue')
                        print(self.arg_queue)
                        args = self.arg_queue.pop()
                        #Process
                        print('this is args')
                        print(args)
                        if args==():
                            result = self.process(entry)
                        else:
                            result = self.process(entry, args)
                        self.output_queue.appendleft(result)
                    
                    #Write the queue results to file
                    while len(self.output_queue) > 0:
                        print(self.bc['class_name'],"writing to files")
                        entry = self.output_queue.pop()
                        
                        #Write to all the files it knows including stubs.
                        for p in self.bc['out_stubs']:
                            self.write_file(entry, p)
                            
                        for p in self.bc['output_files']:
                            self.write_file(entry, p)
                    
                else:
                    print("File not available, can't read in")
            
        else:
            #Wait for change in files that are being listened to.
            
            
            #print(self.bc['class_name'], "watching for file changes")
            
            directory = 'comms_files'

            file_records = {}
            
            while True:
                #print(self.bc['class_name'], "checking for file changes")
                for filename in os.listdir(directory):
                    f = os.path.join(directory, filename)
                    # checking if it is a file
                    if os.path.isfile(f): 
                        modified = False
                        fits_criteria = False
                        
                        path = f
                        
                        if path in file_records:
                            #print("In records")
                            if os.path.getmtime(path) > file_records[path]:
                                file_records[path] = os.path.getmtime(path)
                                
                                print(self.bc['class_name'],"detected file Modified")
                                modified = True
                                    
                        else:
                            print(self.bc['class_name'], "detected new File")
                            file_records[path] = os.path.getmtime(path)
                            
                            #print("File Modified")
                            modified = True
                            
                        #----Process if Modified-----
                        if modified:
                            #Check if it is in the ones the block is listening to.
                            print(self.bc['class_name'],"checking if path is included")
                            print("Is ", path, "included?")
                            
                            if path in self.bc['input_files']:
                                #print("Yes 1")
                                fits_criteria = True
                                
                            for expression in self.bc['in_regex']:
                                print("Expression to match: ", expression)
                                if re.match(expression, path):
                                    #print("Yes 2")
                                    fits_criteria = True
                            
                            if fits_criteria == True:
                                print(self.bc['class_name'], "thinks the file fits criteria")
                            
                                #----Process Method-------------
                                print(self.bc['class_name'], "processing from file")
                                self.read_file(path)
                                self.collect_args()
                                
                                #Process the queue
                                while len(self.input_queue) > 0:
                                    #Pop off an entry
                                    entry = self.input_queue.pop()
                                    args = self.arg_queue.pop()
                                    #Process
                                    if args==():
                                        result = self.process(entry)
                                    else:
                                        print('we are testing args')
                                        print(args)
                                        result = self.process(entry, args)
                                    #result = self.process(entry)
                                    self.output_queue.appendleft(result)
                                
                                #Write the queue results to file
                                while len(self.output_queue) > 0:
                                    print(self.bc['class_name'], "writing to files \n")
                                    entry = self.output_queue.pop()
                                    
                                    #Write to all the files it knows including stubs.
                                    for p in self.bc['out_stubs']:
                                        self.write_file(entry, p)
                                        
                                    for p in self.bc['output_files']:
                                        self.write_file(entry, p)
                                
                #Sleep before checking again.
                time.sleep(1)
        
        '''
        #--------------Process--------------------------
        #Process the queue
        while len(self.input_queue) > 0:
            #Pop off an entry
            entry = self.input_queue.pop()
            #Process
            result = self.process(entry)
            self.output_queue.appendleft(result)
        
        #Write the queue results to file
        while len(self.output_queue) > 0:
            print("Writing to file")
            
            #self.out_stubs)
        '''
        
    def process(self, entry, args=None):
        #The code to actually work with the data (Reimplement custom)
        print("Please implement the process function")
        pass