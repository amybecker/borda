import csv
import pandas as pd
from rcv_functions import *

def vote_tally(ballot_list, candidate_list):
    num_places = max([len(ballot) for ballot in ballot_list])
    cand_tally = {cand:[0]*num_places for cand in candidate_list}
    for ballot in ballot_list:
        for pl in range(len(ballot)):
            cand_tally[ballot[pl]][pl]+=1
    return cand_tally
        

############ Cambridge Data ############

def load_cambridge_vote_data(election_position, council_size, years):

    cands_dict = {}
    prec_rcvs_dict = {}
    prec_vote_tallies_dict = {}
    total_rcvs_dict = {}
    total_vote_tallies_dict = {}

    for year in years:
        print('processing: ', year)
        election_file_name = './Cambridge/'+election_position+'/'+str(year)+'.csv'
        bad_labels = ['WI0', 'W','=','(',')','overvote']

        df = pd.read_csv(election_file_name)
        df['prec'] = df.apply(lambda x: x['ID'].split('-')[0], axis =1)

        precincts = list(df['prec'].unique())
        data_list = df.fillna('').values.tolist()
        ballot_dict = {prec:[] for prec in precincts}
        for row in data_list:
            break_ind = -1
            bad_label = False
            for i in range(1,len(row)):
                #if write in or overvote or empty, delete rest of ballot
                for bl in bad_labels:
                    if bl in row[i]:
                        bad_label = True
                if row[i] == '' or bad_label:
                    break_ind = i
                    break
            ballot_dict[row[-1]].append([item.strip() for item in row[1:break_ind]])

        cands = set()
        for precinct in precincts:
            for ballot in ballot_dict[precinct]:
                cands = cands.union(set(ballot))
        cands = list(cands)

        prec_rcv_ranks ={}
        prec_vote_tallies ={}

        for precinct in precincts:
            print('processing: ', year, precinct)
            prec_vote_tally = vote_tally(ballot_dict[precinct],cands)
            prec_vote_tallies[precinct] = prec_vote_tally
            ballot_list = [[ballot,1] for ballot in ballot_dict[precinct]]
            rcv_outcome = rcv_run(ballot_list, council_size, True, False)
            rcv_outcome[1].reverse()
            extra_cands = list(set(cands).difference(set(rcv_outcome[0])).difference(set(rcv_outcome[1])))
            rcv_rank = rcv_outcome[0]+extra_cands+rcv_outcome[1]
            prec_rcv_ranks[precinct] = rcv_rank
            assert(len(rcv_rank)==len(cands))

        all_ballots = [ballot for sublist in [ballot_dict[prec] for prec in precincts] for ballot in sublist]
        overall_vote_tally = vote_tally(all_ballots,cands)

        rcv_ballots = [[ballot,1] for sublist in [ballot_dict[prec] for prec in precincts] for ballot in sublist]
        overall_rcv_outcome = rcv_run(rcv_ballots, council_size, True, False)
        overall_rcv_outcome[1].reverse()
        overall_rcv_rank = overall_rcv_outcome[0]+overall_rcv_outcome[1]

        cands_dict[year] = cands
        prec_rcvs_dict[year] = prec_rcv_ranks
        prec_vote_tallies_dict[year] = prec_vote_tallies
        total_rcvs_dict[year] = overall_rcv_rank
        total_vote_tallies_dict[year] = overall_vote_tally

    return [cands_dict, prec_rcvs_dict, prec_vote_tallies_dict, total_rcvs_dict, total_vote_tallies_dict]


# [csc_cands, csc_prec_STV_vecs, csc_prec_vote_tallies, csc_STV_vecs, csc_vote_tallies] = load_cambridge_vote_data('school_committee', 6)