from get_fitness import fitness, init_population, desimal_to_binary, binary_to_desimal
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import csv

POPULATION_SIZE = 70
GENOME_LENGTH = 15

def check(population):
    count_keep = 0
    new_population = []

    for chromosome in population:
        if chromosome['erase'] == False:
            count_keep += 1
            new_population.append(chromosome)
        else: continue

    if count_keep == 0 or count_keep == 1 :
        return False
    else:
        print("New population from checking: \n", new_population)
        return new_population


def roulette_wheel(population, ix):
    population_fitness = sum(chromosome['fitness'] for chromosome in population)
    probabilities = [chromosome['fitness'] / population_fitness for chromosome in population]
    inverted_probabilities = [1 / p for p in probabilities if p != 0]
    population_temp = [{'chromosome': chromosome, 'probability': prob} for chromosome, prob in zip(population, inverted_probabilities)]

    print("Population temporary: ", [{'chromosome': p['chromosome'], 'probability': p['probability']} for p in population_temp])

    total_inverted = sum(inverted_probabilities)
    normalized_inverted_probabilities = [p / total_inverted for p in inverted_probabilities]
    
    population_with_inverted_prob = [{'chromosome': chromosome, 'probability': prob} for chromosome, prob in zip(population, normalized_inverted_probabilities)]
   
    print("sorted population with inverted probabilities: ", [{'chromosome': p['chromosome'], 'probability': p['probability']} for p in population_with_inverted_prob])
    
    selected_index = np.random.choice(len(population_with_inverted_prob), p=[p['probability'] for p in population_with_inverted_prob])
   
    while selected_index == ix:
        selected_index = np.random.choice(len(population_with_inverted_prob), p=[p['probability'] for p in population_with_inverted_prob])

    print("selected index: ", selected_index)

    return selected_index, population_with_inverted_prob[selected_index]['chromosome']
def crossover(p1, p2, crossover_rate=0.8):

    if random.random() > crossover_rate:
        return (p1['cappv_binary'], p2['cappv_binary'], p1['ebess_binary'], p2['ebess_binary'])

    max_attempts = 10
    attempts = 0

    while attempts < max_attempts:
        point = random.randint(1, GENOME_LENGTH-1)
        child_cappv_1 = np.concatenate([p1['cappv_binary'][:point], p2['cappv_binary'][point:]])
        child_cappv_2 = np.concatenate([p2['cappv_binary'][:point], p1['cappv_binary'][point:]])

        child_ebess_1 = np.concatenate([p1['ebess_binary'][:point], p2['ebess_binary'][point:]])
        child_ebess_2 = np.concatenate([p2['ebess_binary'][:point], p1['ebess_binary'][point:]])

        desimal_child_cappv_1 = binary_to_desimal(child_cappv_1)
        desimal_child_cappv_2 = binary_to_desimal(child_cappv_2)
        desimal_child_ebess_1 = binary_to_desimal(child_ebess_1)
        desimal_child_ebess_2 = binary_to_desimal(child_ebess_2)

        if all(100 <= x <= 20000 for x in [desimal_child_cappv_1, desimal_child_cappv_2, 
                                           desimal_child_ebess_1, desimal_child_ebess_2]):
            return (child_cappv_1, child_cappv_2, child_ebess_1, child_ebess_2)

        attempts += 1

    return (p1['cappv_binary'], p2['cappv_binary'], p1['ebess_binary'], p2['ebess_binary'])

def mutation(individual, mutation_rate=0.1):
    mutated_individual = individual.copy()

    for i in range(GENOME_LENGTH):
        if random.random() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i]
    
    decimal_value = binary_to_desimal(mutated_individual)
    if 100 <= decimal_value <= 20000:
        return mutated_individual

    return individual

def find_index(population, target_fitness):
    for index, chromosome in enumerate(population):
        if chromosome['fitness'] == target_fitness:
            return index
    return -1

def genetic_algorithm(population_size):
    population = init_population(population_size)
    print("Fitness \n\n")

    population_val = fitness(population)

    i = 1
    iteration = 0
    generations = 50

    fitness_all = []
    cappv_all = []
    cpv_all = []
    ebess_all = []
    cbess_all = []
    total_cost_all = []

    while(iteration < generations):
        print(f"\nGeneration {i}:\n")

        iteration += 1
        result = check(population_val)

        if result is False:
            population_withoutfitness = init_population(POPULATION_SIZE)
            population_val = fitness(population_withoutfitness)
            generations += 1

            print("This generation is being skipped")
        else:
            population_check = result

            new_population = []
            new_population_temp = []

            ix = -1
            ix, p1 = roulette_wheel(population_check, ix)
            ix, p2 = roulette_wheel(population_check, ix)

            print("p1 awal generasi: \n", p1)
            print("p2 awal generasi: \n", p2)
            
            while len(new_population) < population_size // 2:

                print("p1 masuk crossover: \n", p1)
                print("p2 masuk crossover: \n", p2)

                #crossover
                offspring_cappv_1, offspring_cappv_2, offspring_ebess_1, offspring_ebess_2 = crossover(p1, p2)

                #mutation
                offspring_mutation_cappv_1 = mutation(offspring_cappv_1)
                offspring_mutation_cappv_2 = mutation(offspring_cappv_2)
                offspring_mutation_ebess_1 = mutation(offspring_ebess_1)
                offspring_mutation_ebess_2 = mutation(offspring_ebess_2)

                desimal_offspring_mutation_cappv_1 = binary_to_desimal(offspring_mutation_cappv_1)
                desimal_offspring_mutation_cappv_2 = binary_to_desimal(offspring_mutation_cappv_2)
                desimal_offspring_mutation_ebess_1 = binary_to_desimal(offspring_mutation_ebess_1)
                desimal_offspring_mutation_ebess_2 = binary_to_desimal(offspring_mutation_ebess_2)

                new_child_1 = dict(cappv_bits= offspring_mutation_cappv_1, cappv_desimal= desimal_offspring_mutation_cappv_1, ebess_bits= offspring_mutation_ebess_1, ebess_desimal= desimal_offspring_mutation_ebess_1)
                new_child_2 =  dict(cappv_bits= offspring_mutation_cappv_2, cappv_desimal= desimal_offspring_mutation_cappv_2, ebess_bits= offspring_mutation_ebess_2, ebess_desimal= desimal_offspring_mutation_ebess_2)

                new_population_temp.append(new_child_1)
                new_population_temp.append(new_child_2)

                new_population_temp_val = fitness(new_population_temp)
                result = check(new_population_temp_val)

                if result is not False:
                    new_population.append(new_child_1)
                    new_population.append(new_child_2)

                    new_population_val = fitness(new_population)

                    iy = -1
                    iy, p1 = roulette_wheel(new_population_val, iy)
                    iy, p2 = roulette_wheel(new_population_val, iy)

                    print("p1 hasil roulette wheel: \n", p1)
                    print("p2 hasil roulette wheel: \n", p2)

                else:
                    print("p1 tanpa roulette wheel: \n", p1)
                    print("p2 tanpa roulette wheel: \n", p2)

                new_population_temp = []

            population_val = new_population_val
            # population_withoutfitness = new_population
            # print(f"population_withoutfitness: {population_withoutfitness}\n")

            # population_val = fitness(population_withoutfitness)
            
            best_fitness = min(chromosome['fitness'].real for chromosome in population_val)
            index_chromosome, best_chromosome = min(enumerate(population_val), key=lambda x: x[1]['fitness'].real)

            # print("Best fitness value:", best_chromosome['fitness'].real)
            # print("Index of best fitness:", index_chromosome)

            # print(f"Generation: {i}, Best Fitness: {best_chromosome}")

            # === CONVERGENCE LOGIC ===
            # Only store the best chromosome if its fitness is <= previous best.
            # If it's higher, repeat the previous best (so the curve never goes up).
            if len(fitness_all) == 0:
                # first valid generation: just store it
                fitness_all.append(best_chromosome['fitness'])
                cappv_all.append(best_chromosome['cappv_desimal'])
                ebess_all.append(best_chromosome['ebess_desimal'])
                total_cost_all.append(best_chromosome['TC'])
                cpv_all.append(best_chromosome['cpv'])
                cbess_all.append(best_chromosome['cbess'])
            else:
                prev_best_fitness = fitness_all[-1].real if hasattr(fitness_all[-1], 'real') else fitness_all[-1]
                current_best_fitness = best_chromosome['fitness'].real

                if current_best_fitness <= prev_best_fitness:
                    # better or equal -> store the new best
                    fitness_all.append(best_chromosome['fitness'])
                    cappv_all.append(best_chromosome['cappv_desimal'])
                    ebess_all.append(best_chromosome['ebess_desimal'])
                    total_cost_all.append(best_chromosome['TC'])
                    cpv_all.append(best_chromosome['cpv'])
                    cbess_all.append(best_chromosome['cbess'])
                    print(f"Generation {i}: NEW BEST stored. Fitness = {current_best_fitness}")
                else:
                    # worse -> keep previous best (carry it forward)
                    fitness_all.append(fitness_all[-1])
                    cappv_all.append(cappv_all[-1])
                    ebess_all.append(ebess_all[-1])
                    total_cost_all.append(total_cost_all[-1])
                    cpv_all.append(cpv_all[-1])
                    cbess_all.append(cbess_all[-1])
                    print(f"Generation {i}: worse fitness ({current_best_fitness}) > prev best ({prev_best_fitness}). Keeping previous best.")
        
        i += 1
    
    # Use the lowest fitness ever stored (last value of fitness_all is the converged best,
    # since fitness_all is now monotonically non-increasing).
    if len(fitness_all) > 0:
        best_fitness = fitness_all[-1].real if hasattr(fitness_all[-1], 'real') else fitness_all[-1]
    best_index_final = find_index(population_val, best_fitness)
    # print("Best fitness: ", round(population_val[best_index_final]['fitness'].real, 2))
    # print("Best cappv:", round(population_val[best_index_final]['cappv_desimal'].real, 2))
    # print("Best ebess: ", round(population_val[best_index_final]['ebess_desimal'].real, 2))
    # print("Erase: ", population_val[best_index_final]['erase'])
    # print("Best CPV: ", round(population_val[best_index_final]['cpv'].real, 2))
    # print("Best IC: ", round(population_val[best_index_final]['ic'].real, 2))
    # print("Best DC: ", round(population_val[best_index_final]['dc'].real, 2))
    # print("Best CBESS: ", round(population_val[best_index_final]['cbess'].real, 2))
    # print("Best TC: ", round(population_val[best_index_final]['TC'].real, 2))

    # print("\nBest final result: ", population_val[best_index_final])

    return fitness_all, cappv_all, ebess_all, total_cost_all, cpv_all, cbess_all


def do_convergence(
    fitness_all,
    cappv_all,
    ebess_all,
    total_cost_all,
    cpv_all,
    cbess_all,
    filename
):


    # write CSV
    csv_filename = f'ga_convergence_results_{filename}.csv'
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # ✅ FIXED HEADER
        writer.writerow([
            'Iteration',
            'Fitness',
            'CAPPV',
            'CPV',
            'EBESS',
            'CBESS',
            'Total_Cost',

        ])

        # rows
        for i in range(len(fitness_all)):
            writer.writerow([
                i + 1,
                round(fitness_all[i], 4),
                round(cappv_all[i], 4),
                round(cpv_all[i], 4),
                round(ebess_all[i], 4),
                round(cbess_all[i], 4),
                round(total_cost_all[i], 4),
            ])

    return (
        fitness_all,
        cappv_all,
        ebess_all,
        total_cost_all
    )

def make_graph(fitness_all, cappv_all, ebess_all, total_cost_all, count):
    fig, axs = plt.subplots(4, 1, figsize=(15, 12))

    axs[0].plot(fitness_all, marker='o', linestyle='-', markersize=4)
    axs[0].set_title('Fitness Over Time')
    axs[0].set_ylabel('Fitness')
    axs[0].set_xlabel('Time Steps')

    axs[1].plot(cappv_all, marker='o', linestyle='-', markersize=4)
    axs[1].set_title('CAPPV Over Time')
    axs[1].set_ylabel('CAPPV')
    axs[1].set_xlabel('Time Steps')

    axs[2].plot(ebess_all, marker='o', linestyle='-', markersize=4)
    axs[2].set_title('EBESS Over Time')
    axs[2].set_ylabel('EBESS')
    axs[2].set_xlabel('Time Steps')

    axs[3].plot(total_cost_all, marker='o', linestyle='-', markersize=4)
    axs[3].set_title('Total Cost Over Time')
    axs[3].set_ylabel('Total Cost')
    axs[3].set_xlabel('Time Steps')

    plt.tight_layout()

    save_path = f'graph_output{count}.jpg'
    plt.savefig(save_path, format='jpg')
    plt.close(fig)
