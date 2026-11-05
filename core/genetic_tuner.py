"""
Genetic Tuner - Evolutionary Optimization Engine
Inspired by Alpha Evolve / AutoML

This module implements a genetic algorithm to evolve system configuration parameters
over time. It treats configuration settings as "genes" and system performance as "fitness".

Algorithm:
1. Population: A set of configuration profiles (Genome).
2. Mutation: Randomly tweaking parameters (e.g., +/- 10% buffer size).
3. Crossover: Combining best features of top performing profiles.
4. Selection: Keeping the fittest profiles based on a fitness function (FPS/VRAM).

Target:
- Optimizes UE5 Console Variables (Scalability, Memory)
- Optimizes Houdini Environment Variables (Thread counts, Memory limits)
"""

import os
import json
import random
import time
import logging
import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Configure logging
logger = logging.getLogger("GeneticTuner")

@dataclass
class Gene:
    """A single configurable parameter"""
    name: str
    value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: Optional[List[Any]] = None
    type: str = "float"  # float, int, choice

@dataclass
class Genome:
    """A complete configuration profile (DNA)"""
    id: str
    genes: Dict[str, Gene]
    fitness_score: float = 0.0
    generation: int = 0
    parent_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

class GeneticTuner:
    def __init__(self, data_dir: str = "data/genetic_evolution"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.data_dir / "evolution_history.json"
        self.population_file = self.data_dir / "current_population.json"
        
        self.population_size = 5
        self.mutation_rate = 0.2
        self.elite_size = 1
        
        self.current_population: List[Genome] = []
        self._load_state()

    def _load_state(self):
        """Load population from disk"""
        if self.population_file.exists():
            try:
                with open(self.population_file, 'r') as f:
                    data = json.load(f)
                    # Reconstruct Genome objects (simplified)
                    self.current_population = [] 
                    for item in data:
                        genes = {k: Gene(**v) for k, v in item['genes'].items()}
                        self.current_population.append(Genome(
                            id=item['id'],
                            genes=genes,
                            fitness_score=item['fitness_score'],
                            generation=item['generation'],
                            parent_id=item['parent_id'],
                            created_at=item['created_at']
                        ))
                logger.info(f"Loaded population of {len(self.current_population)} genomes")
            except Exception as e:
                logger.error(f"Failed to load population: {e}")
                self.current_population = []

    def _save_state(self):
        """Save population to disk"""
        try:
            data = [asdict(g) for g in self.current_population]
            with open(self.population_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def initialize_population(self, baseline_config: Dict[str, Any]):
        """Create initial population from baseline config"""
        if self.current_population:
            return # Already initialized

        logger.info("Initializing new genetic population...")
        
        # Define the "Gene Pool" - Mutable parameters
        # This maps config keys to Gene definitions
        base_genes = {
            # UE5 Genes
            "r.Streaming.PoolSize": Gene("r.Streaming.PoolSize", 12288, 4096, 20000, type="int"),
            "r.Lumen.ScreenTraces": Gene("r.Lumen.ScreenTraces", 1, 0, 1, type="int"), # Boolean-ish
            "r.Nanite.MaxPixelsPerEdge": Gene("r.Nanite.MaxPixelsPerEdge", 1.0, 0.5, 4.0, type="float"),
            
            # Houdini Genes
            "HOUDINI_PDG_MAX_CONCURRENT_COOKS": Gene("HOUDINI_PDG_MAX_CONCURRENT_COOKS", 24, 4, 48, type="int"),
            "HOUDINI_VDB_FAST_TEMP_FILE": Gene("HOUDINI_VDB_FAST_TEMP_FILE", 1, 0, 1, type="int"),
        }

        # Create Baseline (Adam)
        adam = Genome(
            id="gen0_baseline",
            genes=base_genes,
            generation=0
        )
        self.current_population.append(adam)

        # Create initial mutations
        for i in range(self.population_size - 1):
            mutant = self._mutate(adam, generation=0)
            mutant.id = f"gen0_mutant_{i+1}"
            self.current_population.append(mutant)
            
        self._save_state()

    def _mutate(self, parent: Genome, generation: int) -> Genome:
        """Create a mutated copy of a genome"""
        new_genes = {}
        
        for name, gene in parent.genes.items():
            # Deep copy gene
            new_gene = Gene(**asdict(gene))
            
            # Chance to mutate
            if random.random() < self.mutation_rate:
                if gene.type == "float":
                    # +/- 10% mutation
                    change = new_gene.value * 0.1 * (1 if random.random() > 0.5 else -1)
                    new_gene.value = max(gene.min_value, min(gene.max_value, new_gene.value + change))
                elif gene.type == "int":
                    # +/- 1 or 10%
                    change = max(1, int(new_gene.value * 0.1)) * (1 if random.random() > 0.5 else -1)
                    new_gene.value = max(gene.min_value, min(gene.max_value, int(new_gene.value + change)))
                elif gene.type == "choice":
                    new_gene.value = random.choice(gene.options)
            
            new_genes[name] = new_gene

        return Genome(
            id=f"gen{generation+1}_{int(time.time())}",
            genes=new_genes,
            generation=generation + 1,
            parent_id=parent.id
        )

    def evaluate_fitness(self, genome: Genome) -> float:
        """
        Run benchmarks to determine fitness score.
        
        Fitness = (Performance_Metric / Resource_Cost)
        e.g., FPS / VRAM_Usage
        
        NOTE: This simulates the benchmark for now. 
        In production, this would launch a headless UE5/Houdini benchmark.
        """
        logger.info(f"Evaluating genome {genome.id}...")
        
        # TODO: Implement actual benchmarking harness
        # For now, we simulate a fitness score based on "ideal" values
        # giving higher score to values closer to our known good heuristics
        
        score = 100.0
        
        # Simulation logic: 
        # - Penalize very low streaming pool (stutter)
        # - Penalize very high streaming pool (OOM risk)
        pool_size = genome.genes["r.Streaming.PoolSize"].value
        if pool_size < 6000: score -= 20
        if pool_size > 18000: score -= 10 # Diminishing returns
        
        # - Reward high thread count up to a point
        threads = genome.genes["HOUDINI_PDG_MAX_CONCURRENT_COOKS"].value
        if threads > 32: score -= 5 # Context switching overhead
        if threads < 8: score -= 20 # Underutilization
        
        # Add random noise to simulate real-world variance
        score += random.uniform(-5.0, 5.0)
        
        return max(0.0, score)

    def run_evolution_cycle(self):
        """Run one generation of evolution"""
        logger.info(f"Starting evolution cycle for Generation {self.current_population[0].generation}")
        
        # 1. Evaluate Fitness
        for genome in self.current_population:
            if genome.fitness_score == 0.0: # Only evaluate if not already scored
                genome.fitness_score = self.evaluate_fitness(genome)
        
        # 2. Selection (Sort by fitness)
        self.current_population.sort(key=lambda g: g.fitness_score, reverse=True)
        
        # Log results
        best = self.current_population[0]
        logger.info(f"Best genome: {best.id} (Score: {best.fitness_score:.2f})")
        
        # 3. Create Next Generation
        next_gen = []
        
        # Elitism: Keep the best
        for i in range(self.elite_size):
            next_gen.append(self.current_population[i])
            
        # Reproduction: Mutate the top performers to fill the rest
        while len(next_gen) < self.population_size:
            parent = random.choice(self.current_population[:2]) # Pick from top 2
            child = self._mutate(parent, parent.generation)
            next_gen.append(child)
            
        self.current_population = next_gen
        self._save_state()
        
        return best

    def get_best_config(self) -> Dict[str, Any]:
        """Return the configuration of the fittest genome"""
        if not self.current_population:
            return {}
        
        # Sort just in case
        self.current_population.sort(key=lambda g: g.fitness_score, reverse=True)
        best = self.current_population[0]
        
        return {name: gene.value for name, gene in best.genes.items()}

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    tuner = GeneticTuner()
    tuner.initialize_population({})
    best = tuner.run_evolution_cycle()
    print(f"Optimization complete. Best config: {tuner.get_best_config()}")
