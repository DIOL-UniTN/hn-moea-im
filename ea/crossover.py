import inspyred
from ea.mutation import ea_global_random_mutation

@inspyred.ec.variators.crossover
def ea_crossover(random, candidate1, candidate2, args):
    common = list(set(candidate1).intersection(set(candidate2)))                # see common elements
    max_trials = 5

    # apply mutation while the different genes are less than 2 for max_trials times
    while (len(candidate1) - len(common)) < 2 and max_trials > 0:
        if len(candidate1) - len(common) == 1:                                  # if the two candidates differ by 1 element, perform a random mutation once
            candidate1 = ea_global_random_mutation(random, [candidate1], args)[0]
            candidate2 = ea_global_random_mutation(random, [candidate2], args)[0]
        elif len(candidate1) == len(common):                                    # if the two candidates are identical, perform a random mutation twice
            for _ in range(2):
                candidate1 = ea_global_random_mutation(random, [candidate1], args)[0]
                candidate2 = ea_global_random_mutation(random, [candidate2], args)[0]

        max_trials -= 1
        common = list(set(candidate1).intersection(set(candidate2)))

    if max_trials==0:
        return [candidate2, candidate1]

    # make a copy of the candidates
    new_candidate1 = candidate1.copy()
    new_candidate2 = candidate2.copy()
    c1_common = {}
    c2_common = {}

    # remove all the genes in common from each candidate
    for c in common:
        new_candidate1.pop(new_candidate1.index(c))
        new_candidate2.pop(new_candidate2.index(c))

        # save the indexes of the common genes
        idx1 = candidate1.index(c)
        idx2 = candidate2.index(c)
        c1_common[idx1] = c
        c2_common[idx2] = c

    # choose a swap point
    # if candidates have different lengths it works anyway
    swap_idx = random.randint(1, len(new_candidate1) - 1) # if swap_idx = 0, all the genes are swapped, so no crossover is done
    swap = new_candidate1[swap_idx:]
    new_candidate1[swap_idx:] = new_candidate2[swap_idx:]
    new_candidate2[swap_idx:] = swap

    # reinsert the common genes 
    for (idx, c) in c1_common.items():
        new_candidate1.insert(idx, c) 
    for (idx, c) in c2_common.items():
        new_candidate2.insert(idx, c)

    return [new_candidate1, new_candidate2] 
