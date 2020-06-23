"""

this tool runs simple instances of rank-choice voting.
votes that exceed the threshold are transfered proportionally to 
their remaining share among votes with same first choice.

The rcv_run function takes as input a list of rankings (first choice to last) 
and percent of voters with that ranking:

[['candidate_1', 'candidate 2', ....], percent with this rank choice], 
	['candidate_2', 'candidate 1', ....], percent with this rank choice],
	...]

e.g.:
	ballot_list = [[['a1','a2','b1'], 0.5],
				[['a1','b1','a2'], .1],
				[['b1','a1'], .4]]


percentages need not sum to one and can optionally be rescaled

rankings need not be complete

"""

import math
import random

def remove_cand(cand, ballot_list):
	for ballot in ballot_list:
		new_ballot = []
		for c in ballot[0]:
			if c!= cand:
				new_ballot.append(c)
		ballot[0] = new_ballot


def replace_cand(cand_rem, cand_add, ballot_list):
	for ballot in ballot_list:
		new_ballot = []
		for c in ballot[0]:
			if c== cand_rem:
				new_ballot.append(cand_add)
			else:
				new_ballot.append(c)
		ballot[0] = new_ballot


def transfer_votes(cand, ballot_list, cutoff):
	winning_ballots = []
	winning_vote_share = 0
	for i in range(len(ballot_list)):
		if len(ballot_list[i][0]) == 0:
			continue
		if ballot_list[i][0][0] == cand:
			winning_ballots.append(i)
			winning_vote_share += ballot_list[i][1]
	for i in range(len(ballot_list)):
		if i in winning_ballots:
			ballot_list[i][1] = ballot_list[i][1] - cutoff*ballot_list[i][1]/winning_vote_share



def rcv_run(ballot_list, num_seats, rescale_bool, verbose_bool):
	#rescale
	rescale_val = 1
	if rescale_bool:
		rescale_val = 0
	for ballot in ballot_list:
		rescale_val += ballot[1]
	if rescale_val == 0:
		print("need to have positive ballot percentages")
	for ballot in ballot_list:
		ballot[1] = ballot[1]/rescale_val

	winners = []
	losers = []
	cutoff = 1.0/(num_seats+1)

	#identify winners
	while len(winners) < num_seats:
		# print(ballot_list)
		# print()
		max_list_len = max(len(ballot[0]) for ballot in ballot_list)
		if max_list_len == 0:
			print("******* WARNING: threshold not passed *******")
			break
		cand_dict = {ballot[0][0]:0 for ballot in ballot_list if len(ballot[0]) > 0}
		for ballot in ballot_list:
			if len(ballot[0]) > 0:
				cand_dict[ballot[0][0]] += ballot[1]
		max_cand = max(cand_dict, key=cand_dict.get)
		min_cand = min(cand_dict, key=cand_dict.get)
		if cand_dict[max_cand] >= cutoff:
			winners.append(max_cand)
			transfer_votes(max_cand, ballot_list, cutoff)
			remove_cand(max_cand, ballot_list)
			if verbose_bool:
				print("candidate", max_cand, "elected")
		elif cand_dict[min_cand] < cutoff:
			remove_cand(min_cand, ballot_list)
			losers.append(min_cand)
			if verbose_bool:
				print("candidate", min_cand, "eliminated")
				
		else:
			print("******* ISSUE: input error *******")
			break
	return winners, losers


