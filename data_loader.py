import os
from os import listdir
from os.path import isfile, join
import numpy as np

from tqdm import trange
from tqdm import tqdm




#Load the data of Task 1
class Data_Loader:
    def __init__(self,options):
        """Obtain the path of raw data."""
        positive_data_file = options['dir_name']
        
        
        
        """
        Obtain the path of the "item index dictionary". (item or search query).
        Data_Loader generates the "item index dictionary" and saves it to the "dir_name_index" paths (i.e., index_data_file).
        """
        index_data_file = options['dir_name_index']
        
        
        
        """Read datasets"""
        print("Read datasets")
        positive_examples = list(open(positive_data_file,'r').readlines())
        positive_examples = [s for s in tqdm(positive_examples)]#positive_examples form e.g.,: [["0,0,1,4"],["0,0,3,5"],["0,3,4,9"], ...]
        x_list = [x[:-1].split(",") for x in tqdm(positive_examples)] #x_list form e.g.,: [["0","0","1","4"],["0","0","3","5"],["0","3","4","9"], ...]
        
        
        
        """
        Remove duplicated items to generate an intrinsic item index for each item.
        """
        print("Obtain the unique items")
        unique_ = set()
        for x in tqdm(x_list):
            unique_.update(x)



        """
        Generate an "item index dictionary" where the item indices correspond to the intrinsic index for each item.
        The index 0 signifies the padding of the item sequence.
        """
        print("Generate the 'item index dictionary'")
        self.item_dict = {'0':0} #The index 0 signifies the padding of the item sequence.
        i = 0
        for k in tqdm(sorted(unique_,key=int)):
            self.item_dict[k] = i #Give index to the item.
            i+=1
        
        
        
        """
        Map the raw data to their respective item indices using the "item index dictionary".
        Therefore, the "self.item" consists of a sequence of item indices for each user.
        """
        print("Map the raw data to their respective item indices")
        self.item = np.array([[self.item_dict[x] for x in xs] for xs in tqdm(x_list)])
        self.embed_len = len(self.item_dict) #Get the total number of unique items.



        """Save the "item index dictionary" to the "dir_name_index" path. (i.e., index_data_file)"""
        f = open(index_data_file,'w')
        f.write(str(self.item_dict))
        f.close()
        print("The index has been written to {}".format(index_data_file))



#Load the data of Task 2, 3, ... 
class Data_Loader_Sup:
    def __init__(self,options):
        """Obtain the path of raw data."""
        positive_data_file = options['dir_name']



        """
        Obtain the path of the "item index dictionary" that was generated in Task 1.
        Load the "item index dicxtionary."
        """
        index_data_file = options['dir_name_index']
        self.item_dict = self.read_dict(index_data_file) #Load the "item index dictionary."



        """
        Read datasets.
        Seperate the source (inputs) and target (labels) of Task. The source and target are split by ",,".
        """
        print("Read datasets")
        positive_examples = list(open(positive_data_file, "r").readlines()) #positive_examples form e.g.,: [["0,0,1,4,,0\n"],["0,0,3,5,,3\n"],["0,3,4,9,,7,4,3\n"], ...]
        colon = ",," #",," is the split separator. 
        source = [s[:-1].split(colon)[0] for s in tqdm(positive_examples)] #The source is the input for models, representing the users' behavior sequence as in Task 1.
        target = [s[:-1].split(colon)[1] for s in tqdm(positive_examples)] #Target is the label of Task.



        """
        Map the raw data to their respective item indices using the "item index dictionary".
        Therefore, the "self.item" consists of a sequence of item indices for each user.
        """
        print("Map the raw data to their respective item indices")
        self.item = self.map_dict(self.item_dict, source)
        self.item_seq_len = self.item.shape[1] #Get the length of the item sequence (i.e., behavior sequence).

        
        
        """
        Remove duplicated targets to generate an intrinsic item index for each target. (target = label)
        This process is the same as obtaining the "item index dictionary" in the Data_Loader function of Task 1.
        The only difference is that this process is performed on the target instead of the source.
        """
        print("Obtain the unique targets")
        x_list = [x.split(",") for x in target]
        unique_ = set()
        for x in x_list:
            unique_.update(x)



        """
        Generate an "target index dictionary" where the target indices correspond to the intrinsic index for each target.
        This process is the same as obtaining the "item index dictionary" in the Data_Loader function of Task 1.
        The only difference is that this process is performed on the target instead of the source.
        """
        self.target_dict = {}
        i = 0
        for k in tqdm(sorted(unique_,key=int)):
            self.target_dict[k] = i
            i+=1
        self.embed_len = len(self.item_dict) #Get the total number of unique items of source.
        
        
        
        """
        Combine the source and target as [source sequence, target] e.g., [[0,0,1,3,0],[0,0,2,7,1],[0,1,4,5,4], ...]
        If a user has multiple targets, for example, source: 0,3,4,9 and targets: 7,4,3, then we split the targets independently as follows:
        soucre: 0,3,4,9 and targets: 7,4,3 --> [0,3,4,9,7], [0,3,4,9,4], [0,3,4,9,3]
        """
        print("Map the target to their respective indices and prepare the datasets")
        lens = self.item.shape[0] #The total number of sequences (i.e., users).
        self.example = []
        for line in trange(lens):
            source_line = self.item[line]
            target_line = x_list[line]
            for t in target_line: #split the targets independently
                unit = np.append(source_line, np.array(self.target_dict[t]))
                self.example.append(unit)
                
        self.example = np.array(self.example) #The self.example is the processed dataset for train/valid/test.



    """
    read_dict() function reads the item index dictionary
    """
    def read_dict(self, index_data_file):
        dict_temp = {}
        file = open(index_data_file, 'r')
        for line in file.readlines():
            dict_temp = eval(line)
        return dict_temp



    """
    map_dict() function maps the "source" (i.e., raw data) to their respective item indices using the "dict_pretrain" (i.e., "item index dictionary")
    """
    def map_dict(self, dict_pretrain, source):
        items = []
        for lines in tqdm(source):
            trueline = [dict_pretrain[x] for x in lines.split(',')]
            trueline = np.array(trueline)
            items.append(trueline)
        return np.array(items)