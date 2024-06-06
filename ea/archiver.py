def ea_archiver(random, population, archive, args):
    new_archive = archive
    for ind in population:        
        if len(new_archive) == 0:
            new_archive.append(ind)
        else:
            should_remove = []
            should_add = True
            for a in new_archive:
                if ind.candidate == a.candidate and ind.fitness == a.fitness:
                    should_add = False
                    break               
                elif ind < a:
                    should_add = False
                elif ind > a:
                    should_remove.append(a)
            for r in should_remove:                
                new_archive.remove(r)
            if should_add:
                new_archive.append(ind)
    return new_archive