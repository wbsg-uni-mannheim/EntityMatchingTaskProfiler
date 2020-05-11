import random
import csv
from IPython.display import Markdown, display
import pandas as pd
from datautils import*
from similarityutils import*
import re

class GoldStandard(object):
    
    def __init__(self,pos_threshold_, neg_threshold_, posnegratio_, ratioforoneentity_,directory_, sourcea, sourceb, label_att):
        self.pos_threshold = pos_threshold_
        self.neg_threshold = neg_threshold_
        self.posnegratio = posnegratio_
        self.ratioforoneentity = ratioforoneentity_
        self.directory = directory_
        self.sourceb = sourceb
        self.sourcea= sourcea
        self.label_att = label_att
    
   
    def getSimilarityScore(self, ce, sourceb_entity, sourcea_entity):
        try:
            text_sourcea = ''
            text_sourceb = ''
            for common_element in ce:
                if str(sourceb_entity[common_element])=='nan' or str(sourcea_entity[common_element])=='nan':
                    return -1.0
                elif (type(sourceb_entity[common_element]) is str):
                    sourceb_values = str(sourceb_entity[common_element])
                    sourcea_values = str(sourcea_entity[common_element])
                else: 
                    sourceb_values = str(sourceb_entity[common_element].values[0])
                    sourcea_values = str(sourcea_entity[common_element].values[0])

                if (sourceb_values):
                    text_sourceb = text_sourceb + ' ' + sourceb_values
                if (sourcea_values):
                    text_sourcea = text_sourcea + ' ' + sourcea_values

            text_sourceb_processed = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(text_sourceb.lower())).strip()
            text_sourcea_processed = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(text_sourcea.lower())).strip()

            sim_score = get_relaxed_jaccard_sim(text_sourceb_processed,text_sourcea_processed)
            return sim_score
        except:
            print("Exception while calculating similarities")
            import pdb; pdb.set_trace();
            return None

    def goldStandardCreation(self, completeGS=False):

            source_a = pd.read_csv('%s/%s.csv'  % (self.directory, self.sourcea), sep=',')
            source_a['subject_id'] = source_a['subject_id'].astype(str) 
            #this added because the ids are not aligned
            source_a['subject_id'] = source_a['subject_id'].str.lower()
            source_b = pd.read_csv('%s/%s.csv'  % (self.directory, self.sourceb), sep=',')
            source_b['subject_id'] = source_b['subject_id'].astype(str)
            source_b['subject_id'] = source_b['subject_id'].str.lower()

            gs_source = pd.read_csv('%s/goldstandard_true.csv'  % (self.directory), sep=',') 
            gs_source['source_id'] = gs_source['source_id'].astype(str)
            gs_source['target_id'] = gs_source['target_id'].astype(str)
            gs_source['source_id'] = gs_source['source_id'].str.lower()
            gs_source['target_id'] = gs_source['target_id'].str.lower()

            hardpositives = []
            hardnegatives = []
            randompositives = []
            randomnegatives = []

            if completeGS:
                for index,gs_e in gs_source.iterrows():
                    sourcea_id = gs_e['source_id']
                    sourceb_id = gs_e['target_id']
                    matching = gs_e['matching']
                    if matching == 'TRUE' or matching =='true' or matching ==1:
                        randompositives.append(sourcea_id+';'+sourceb_id+';'+'true')
                    elif matching == 'FALSE' or matching =='false' or matching ==0:
                        randomnegatives.append(sourcea_id+';'+sourceb_id+';'+'false')
                    else: print("Cannot parse matching value: ", matching)
                results = {
                    "hardpositives":hardpositives,
                    "hardnegatives":hardnegatives,
                    "randompositives":randompositives,
                    "randomnegatives":randomnegatives
                }

                return results

            #select hard positives
            if self.pos_threshold>1.0:
                print("All positives will be added in the final gold standard. This is oly valid if the GS contains *ONLY* matching pairs.")
                for index,gs_e in gs_source.iterrows():
                    if (index % 100000 ==0): print("Loaded "+str(index)+" matches from GS")
                    sourcea_id = gs_e['source_id']
                    sourceb_id = gs_e['target_id']
                    hardpositives.append(sourcea_id+';'+sourceb_id+';'+'true')
                print("\n %i hard positives were selected" % len(hardpositives))
            else:
                print("Select hard positives")
                i=0
                for index,gs_e in gs_source.iterrows():
                    print_progress(i + 1, len(gs_source), prefix = 'Loading', suffix = 'Complete')
                    i = i+1

                    sourcea_id = gs_e['source_id']
                    sourceb_id = gs_e['target_id']
                    if (not sourcea_id or not sourceb_id): continue
                    sourcea_entity = source_a[source_a['subject_id']==sourcea_id]
                    sourceb_entity = source_b[source_b['subject_id']==sourceb_id]

                    if (sourcea_entity.empty or sourceb_entity.empty): continue
                    if (self.getSimilarityScore(self.label_att, sourcea_entity, sourceb_entity)<self.pos_threshold):
                        hardpositives.append(sourcea_id+';'+sourceb_id+';'+'true')


                print("\n %i hard positives were selected" % len(hardpositives))

                #shuffle the gs and select random positives
                print("Select random positives")
                random_pos_size = int(len(hardpositives)/4)
                i=0
                gs_source.sample(frac=1)
                for index,gs_e in gs_source.iterrows():
                    if (random_pos_size == 0): break
                    sourcea_id = gs_e['source_id']
                    sourceb_id = gs_e['target_id']

                    if (not sourcea_id or not sourceb_id): continue

                    sourcea_entity = source_a[source_a['subject_id']==sourcea_id]
                    sourceb_entity = source_b[source_b['subject_id']==sourceb_id]

                    if (sourcea_entity.empty or sourceb_entity.empty): continue

                    if ( (sourcea_id+';'+sourceb_id+';'+'true') not in hardpositives):
                        random_pos_size = random_pos_size-1
                        randompositives.append(sourcea_id+';'+sourceb_id+';'+'true')
                        print_progress(i + 1, int(len(hardpositives)/4), prefix = 'Loading', suffix = 'Complete')
                        i = i+1


                print("\n %i random positives were selected" % len(randompositives))
            #hard negatives
            hard_neg_size = int(self.posnegratio*len(hardpositives))
            print("Select hard negatives")
            i=0
            for index,sourcea_e in source_a.iterrows():
                if (hard_neg_size == 0): break
                for index_,sourceb_e in source_b.iterrows():
                    try:
                        if (hard_neg_size == 0): break
                        sourcea_id = sourcea_e['subject_id']
                        sourceb_id = sourceb_e['subject_id']
                        # both ids need to exist in the gold standard as we do not know if the datasets are duplicate free
                        if (sourcea_id not in gs_source['source_id'].values or sourceb_id not in gs_source['target_id'].values):
                            continue
                        if (sourcea_id in gs_source['source_id'].values):
                            e = gs_source[gs_source['source_id'] == sourcea_id]
                            if (sourceb_id in e['target_id'].values): continue
                        score_=self.getSimilarityScore(self.label_att, sourcea_e, sourceb_e)
                        if (score_>self.neg_threshold or score_==None or score_==-1):
                            hardnegatives.append(sourcea_id+';'+sourceb_id+';'+'false')
                            hard_neg_size=hard_neg_size-1
                            i = i+1
                            print_progress(i + 1, self.posnegratio*len(hardpositives), prefix = 'Loading', suffix = 'Complete')
                    except: 
                        import pdb; pdb.set_trace();
            print("\n %i hard negatives were selected" % len(hardnegatives))

            #shuffle the sources and select random negative cases
            print("Select random negatives")
            #random_neg_size = int(len(hardnegatives)/4)
            random_neg_size= int(self.posnegratio*len(hardpositives)/4)
            i=0
            source_a = source_a.sample(frac=1)

            for index,sourcea_e in source_a.iterrows():
                if random_neg_size == 0: break
                ratioforoneentity_ = 0
                source_b = source_b.sample(frac=1)
                for index_,sourceb_e in source_b.iterrows():
                    try:
                        if random_neg_size == 0: break
                        #we don't consider if we don't know the corresponding positive as the datasets might be incomplete
                        sourcea_id = sourcea_e['subject_id']
                        sourceb_id = sourceb_e['subject_id']
                        if (sourcea_id not in gs_source['source_id'].values or sourceb_id not in gs_source['target_id'].values):
                            continue
                        if (sourcea_id in gs_source['source_id'].values):
                            e = gs_source[gs_source['source_id'] == sourcea_id]
                            if (sourceb_id in e['target_id'].values): continue

                        if ((sourcea_id+';'+sourceb_id+';'+'false') in hardnegatives): continue
                        i = i+1
                        random_neg_size = random_neg_size-1
                        randomnegatives.append(sourcea_id+';'+sourceb_id+';'+'false')
                        print_progress(i + 1, len(hardnegatives)/4, prefix = 'Loading', suffix = 'Complete')
                        ratioforoneentity_ = ratioforoneentity_+1
                        if ratioforoneentity_==self.ratioforoneentity: break
                    except:
                        import pdb; pdb.set_trace();
            print("\n %i random negatives were selected" % len(randomnegatives))

            self.all_entries = {
                "hardpositives":hardpositives,
                "hardnegatives":hardnegatives,
                "randompositives":randompositives,
                "randomnegatives":randomnegatives
            }

        
    def writeGS(self):
        #add all results into one list
        gs_entries = self.all_entries["hardpositives"]+self.all_entries["hardnegatives"]+self.all_entries["randompositives"]+self.all_entries["randomnegatives"]
        random.shuffle(gs_entries)

        with open(("%s/gold_standard_%s.csv" %(self.directory, self.sourcea)),'w') as f:
            f.write('source_id;target_id;matching\n')
            for gs_entry in gs_entries:
                f.write("%s\n" % gs_entry)