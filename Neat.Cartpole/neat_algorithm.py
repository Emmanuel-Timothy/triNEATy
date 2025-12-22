import random
import numpy as np

# Global Configuration Parameters
INPUT_NODES = 4  # Inputs: Position, Velocity, Angle, Angular Velocity
OUTPUT_NODES = 1 # Output: Force Control (< 0.5 is Left, >= 0.5 is Right)
POPULATION_SIZE = 50
COMPATIBILITY_THRESHOLD = 3.0
C1 = 1.0 # Coefficient for excess genes in compatibility calculation
C2 = 1.0 # Coefficient for disjoint genes in compatibility calculation
C3 = 0.4 # Coefficient for weight differences in compatibility calculation

class NodeGene:
    def __init__(self, node_id, node_type, layer=0):
        self.id = node_id
        self.type = node_type # Node type can be 'INPUT', 'HIDDEN', or 'OUTPUT'
        self.layer = layer # Layer index used for visualization and feed-forward execution order

    def copy(self):
        return NodeGene(self.id, self.type, self.layer)

class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled, innovation):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

    def copy(self):
        return ConnectionGene(self.in_node, self.out_node, self.weight, self.enabled, self.innovation)

class Genome:
    def __init__(self):
        self.nodes = [] # ID -> NodeGene
        self.connections = [] # ConnectionGene list
        self.fitness = 0.0
        self.adjusted_fitness = 0.0

    def add_node(self, node):
        self.nodes.append(node)

    def add_connection(self, connection):
        self.connections.append(connection)

    def mutate(self, innovation_history):
        # Apply weight mutation with 80% probability
        if random.random() < 0.8:
            for conn in self.connections:
                if random.random() < 0.1: # Perturb existing weight
                    conn.weight += random.gauss(0, 0.5)
                elif random.random() < 0.9: # Replace with new random weight
                    conn.weight = random.uniform(-2, 2)
        
        # Apply node addition mutation with 50% probability (Increased)
        if random.random() < 0.5:
            self.mutate_add_node(innovation_history)

        # Apply link addition mutation with 30% probability (Increased)
        if random.random() < 0.3:
            self.mutate_add_link(innovation_history)

    def mutate_add_link(self, innovation_history):
        if len(self.nodes) < 2: return

        # Try 100 times to find a valid link
        for _ in range(100):
            node1 = random.choice(self.nodes)
            node2 = random.choice(self.nodes)

            if node1.layer == node2.layer: continue # Prevent connections within the same layer to maintain simple feed-forward structure
            
            # Enforce feed-forward direction: always connect from a lower layer to a higher layer
            source, target = (node1, node2) if node1.layer < node2.layer else (node2, node1)

            # Check if exists
            exists = False
            for conn in self.connections:
                if conn.in_node == source.id and conn.out_node == target.id:
                    exists = True
                    break
            if exists: continue

            # Create new connection
            innov = innovation_history.get_innovation(source.id, target.id)
            new_conn = ConnectionGene(source.id, target.id, random.uniform(-1, 1), True, innov)
            self.connections.append(new_conn)
            return

    def mutate_add_node(self, innovation_history):
        if not self.connections: return
        
        # Pick a random enabled connection to split
        valid_conns = [c for c in self.connections if c.enabled]
        if not valid_conns: return
        
        conn = random.choice(valid_conns)
        conn.enabled = False # Disable the existing connection

        # Create new node
        # innovation_history also tracks node IDs
        new_node_id = innovation_history.get_node_id()
        
        # Determine the layer of the new node
        in_node = next(n for n in self.nodes if n.id == conn.in_node)
        out_node = next(n for n in self.nodes if n.id == conn.out_node)
        
        # Tidy Layering: Try to reuse existing layers first
        possible_layers = []
        all_layers = sorted(list(set(n.layer for n in self.nodes)))
        for l in all_layers:
            if l > in_node.layer and l < out_node.layer:
                possible_layers.append(l)
        
        if possible_layers:
            new_node_layer = random.choice(possible_layers)
        else:
            new_node_layer = (in_node.layer + out_node.layer) / 2.0

        new_node = NodeGene(new_node_id, 'HIDDEN', new_node_layer)
        self.nodes.append(new_node)

        # Create two new connections
        # 1. Create connection from Input to New Node (weight set to 1.0)
        innov1 = innovation_history.get_innovation(conn.in_node, new_node.id)
        conn1 = ConnectionGene(conn.in_node, new_node.id, 1.0, True, innov1)
        self.connections.append(conn1)

        # 2. Create connection from New Node to Output (inheriting old weight)
        innov2 = innovation_history.get_innovation(new_node.id, conn.out_node)
        conn2 = ConnectionGene(new_node.id, conn.out_node, conn.weight, True, innov2)
        self.connections.append(conn2)

    def distance(self, other):
        # Compute the compatibility distance between two genomes
        # Factors: Excess genes, Disjoint genes, and Average Weight Difference
        # Simplified implementation
        
        innovs1 = {c.innovation: c for c in self.connections}
        innovs2 = {c.innovation: c for c in other.connections}

        all_innovs = set(innovs1.keys()) | set(innovs2.keys())
        
        disjoint = 0
        excess = 0
        weight_diff_sum = 0
        matching = 0

        max_innov1 = max(innovs1.keys()) if innovs1 else 0
        max_innov2 = max(innovs2.keys()) if innovs2 else 0
        max_innov_global = max(max_innov1, max_innov2)

        for i in all_innovs:
            in1 = i in innovs1
            in2 = i in innovs2
            
            if in1 and in2:
                matching += 1
                weight_diff_sum += abs(innovs1[i].weight - innovs2[i].weight)
            elif in1:
                if i > max_innov2: excess += 1
                else: disjoint += 1
            elif in2:
                if i > max_innov1: excess += 1
                else: disjoint += 1

        N = max(len(self.connections), len(other.connections))
        if N < 20: N = 1 # Normalize for large genomes only
        
        weight_diff = weight_diff_sum / matching if matching > 0 else 100

        return (C1 * excess / N) + (C2 * disjoint / N) + (C3 * weight_diff)
    
    @staticmethod
    def crossover(parent1, parent2):
        # Ensure Parent 1 is the fitter genome
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1

        child = Genome()
        for node in parent1.nodes:
            child.add_node(node.copy())
            
        innovs2 = {c.innovation: c for c in parent2.connections}
        
        for conn1 in parent1.connections:
            # Inherit matching genes randomly; otherwise, inherit from the fitter parent
            if conn1.innovation in innovs2:
                if random.random() < 0.5:
                    child.add_connection(conn1.copy())
                else:
                    child.add_connection(innovs2[conn1.innovation].copy())
            else:
                child.add_connection(conn1.copy())
                
        return child

class NeuralNetwork:
    def __init__(self, genome):
        self.genome = genome
        # Sort nodes by layer for feed forward
        self.nodes = sorted(genome.nodes, key=lambda n: n.layer)
        self.node_map = {n.id: n for n in self.nodes}
        self.values = {n.id: 0.0 for n in self.nodes}

    def activate(self, inputs):
        # Set inputs
        input_nodes = [n for n in self.nodes if n.type == 'INPUT']
        for i, node in enumerate(input_nodes):
            if i < len(inputs):
                self.values[node.id] = inputs[i]

        # Feed forward
        for node in self.nodes:
            if node.type == 'INPUT': continue
            
            sum_val = 0.0
            incoming = [c for c in self.genome.connections if c.out_node == node.id and c.enabled]
            for conn in incoming:
                sum_val += self.values[conn.in_node] * conn.weight
            
            self.values[node.id] = self.sigmoid(sum_val)

        # Get output
        output_nodes = [n for n in self.nodes if n.type == 'OUTPUT']
        return [self.values[n.id] for n in output_nodes]

    @staticmethod
    def sigmoid(x):
        try:
            return 1 / (1 + math.exp(-4.9 * x))
        except OverflowError:
            return 0 if x < 0 else 1

import math

class InnovationHistory:
    def __init__(self):
        self.innovations = [] # Tracks innovation numbers for connections defined by (in_node, out_node)
        self.global_innov = 0
        self.global_node = OUTPUT_NODES + INPUT_NODES + 10 # Buffer

    def get_innovation(self, in_node, out_node):
        for i, (inn, out, num) in enumerate(self.innovations):
            if inn == in_node and out == out_node:
                return num
        
        self.global_innov += 1
        self.innovations.append((in_node, out_node, self.global_innov))
        return self.global_innov

    def get_node_id(self):
        self.global_node += 1
        return self.global_node

class Species:
    def __init__(self, representative):
        self.representative = representative
        self.members = [representative]
        self.average_fitness = 0.0
        self.staleness = 0

    def add(self, genome):
        self.members.append(genome)

    def sort_members(self):
        self.members.sort(key=lambda g: g.fitness, reverse=True)
        if self.members[0].fitness > self.representative.fitness:
            self.representative = self.members[0]
            self.staleness = 0
        else:
            self.staleness += 1

    def cull(self):
        if len(self.members) > 2:
            self.members = self.members[:len(self.members)//2]

    def reset(self):
        self.representative = random.choice(self.members)
        self.members = []

class Population:
    def __init__(self, size=POPULATION_SIZE):
        self.size = size
        self.genomes = []
        self.species = []
        self.innovation_history = InnovationHistory()
        self.generation = 1
        
        # Initialize basic genomes
        for _ in range(size):
            g = Genome()
            
            # --- Input Nodes (Layer 0) ---
            input_ids = []
            for i in range(INPUT_NODES):
                g.add_node(NodeGene(i, 'INPUT', 0.0))
                input_ids.append(i)
                
            # --- Output Nodes (Layer 1) ---
            output_ids = []
            for i in range(OUTPUT_NODES):
                oid = INPUT_NODES + i
                g.add_node(NodeGene(oid, 'OUTPUT', 1.0))
                output_ids.append(oid)
            
            # --- Stacked Hidden Layers (Layer 0.33, 0.66) ---
            layer1_ids = []
            layer2_ids = []
            
            # Layer 1
            for _ in range(2): 
                nid = self.innovation_history.get_node_id()
                g.add_node(NodeGene(nid, 'HIDDEN', 0.33))
                layer1_ids.append(nid)
                
            # Layer 2
            for _ in range(2):
                nid = self.innovation_history.get_node_id()
                g.add_node(NodeGene(nid, 'HIDDEN', 0.66))
                layer2_ids.append(nid)
            
            # --- Connections ---
            # 1. Inputs -> Layer 1
            for i in input_ids:
                for h in layer1_ids:
                    innov = self.innovation_history.get_innovation(i, h)
                    g.add_connection(ConnectionGene(i, h, random.uniform(-1, 1), True, innov))

            # 2. Layer 1 -> Layer 2
            for h1 in layer1_ids:
                for h2 in layer2_ids:
                    innov = self.innovation_history.get_innovation(h1, h2)
                    g.add_connection(ConnectionGene(h1, h2, random.uniform(-1, 1), True, innov))
                    
            # 3. Layer 2 -> Outputs
            for h2 in layer2_ids:
                for o in output_ids:
                    innov = self.innovation_history.get_innovation(h2, o)
                    g.add_connection(ConnectionGene(h2, o, random.uniform(-1, 1), True, innov))

            # 4. Residuals: Inputs -> Layer 2 (50% chance)
            for i in input_ids:
                 for h in layer2_ids:
                    if random.random() < 0.5:
                        innov = self.innovation_history.get_innovation(i, h)
                        g.add_connection(ConnectionGene(i, h, random.uniform(-1, 1), True, innov))
            
            self.genomes.append(g)
    
    def run_generation(self):
        # Speciate
        for s in self.species:
            s.reset()
        
        for g in self.genomes:
            found = False
            for s in self.species:
                if g.distance(s.representative) < COMPATIBILITY_THRESHOLD:
                    s.add(g)
                    found = True
                    break
            if not found:
                self.species.append(Species(g))
        
        # Remove empty species
        self.species = [s for s in self.species if len(s.members) > 0]

        # Calculate adjusted fitness
        for s in self.species:
            s.sort_members()
            # Sharing fitness
            for g in s.members:
                g.adjusted_fitness = g.fitness / len(s.members)
        
        # Eliminate stale species and weak species
        
        new_genomes = []
        
        # Elitism: Preserve the best genome of each species if species size exceeds 5
        global_best = max(self.genomes, key=lambda g: g.fitness)
        new_genomes.append(global_best) 
        
        # Calculate total adjusted fitness for roulette wheel
        total_adjusted_fitness = sum(g.adjusted_fitness for g in self.genomes)
        
        # Reproduction phase
        while len(new_genomes) < self.size:
            # Select species
            # Select parent 1
            p1 = self.select_genome(total_adjusted_fitness)
            p2 = self.select_genome(total_adjusted_fitness)
            
            child = Genome.crossover(p1, p2)
            child.mutate(self.innovation_history)
            new_genomes.append(child)
            
        self.genomes = new_genomes
        self.generation += 1
        
    def select_genome(self, total_fitness):
        if total_fitness == 0:
            return random.choice(self.genomes)
            
        r = random.uniform(0, total_fitness)
        cum = 0
        for g in self.genomes:
            cum += g.adjusted_fitness
            if cum >= r:
                return g
        return self.genomes[-1]
